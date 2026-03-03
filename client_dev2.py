import zmq
import time
import numpy as np
import cv2
import torch
import os
from collections import deque

import sim
# --- CONFIGURATION ---
GPU_SERVER_IP = "host.docker.internal"  # IP of your GPU machine
DIFF_PORT = 5555                # Diffusion Policy Port
DINO_PORT = 5556                # Dino World Model Port

IMG_W, IMG_H = 320, 240
CONTROL_HZ = 10
NUM_HIST = 2                    # Context history length
OUTPUT_DIR = "pg"      # Folder to save comparison images
PLAN_BATCH_SIZE = 5    # 'b' - The total number of trajectories to evaluate
# ---------------------

def process_gt_image(img, target_size=224):
    """
    Mimics PyTorch's transforms.Resize(224) + transforms.CenterCrop(224) using OpenCV.
    This prevents aspect ratio squishing so the GT image aligns perfectly with the model's prediction.
    """
    h, w = img.shape[:2]
    
    # 1. Resize shortest side to target_size (aspect ratio is maintained)
    scale = target_size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Center Crop to perfect square
    start_x = (new_w - target_size) // 2
    start_y = (new_h - target_size) // 2
    img_cropped = img_resized[start_y:start_y+target_size, start_x:start_x+target_size]
    
    return img_cropped


class DeviatorAgent():
    def __init__(self, act_seq, b, variance=0.0005):
        self.act_seq = act_seq # T x 4
        self.b = b
        self.std_dev = 5e-3
        
    def perturbe(self):
        # 1. Tile the original sequence 'b' times -> Shape: (b, T, 4)
        A = np.tile(self.act_seq, (self.b, 1, 1))
        
        # 2. Add noise to the remaining (b-1) trajectories
        if self.b > 1:
            T = self.act_seq.shape[0]
            
            # Generate Gaussian noise ONLY for the 3 spatial dimensions (X, Y, Z)
            noise = np.random.normal(0, self.std_dev, size=(self.b - 1, T, 3))
            
            # Apply noise to batches 1 through b, dimensions 0 through 2
            # Gripper (dimension 3) remains identical to the original plan
            A[1:, :, :3] += noise
            
        return A.astype(np.float32)

def main():
    # 1. Setup ZMQ Sockets
    context = zmq.Context()
    
    # Diffusion Policy Socket
    diff_socket = context.socket(zmq.REQ)
    diff_socket.setsockopt(zmq.LINGER, 0) 
    diff_socket.setsockopt(zmq.REQ_RELAXED, 1)
    diff_socket.connect(f"tcp://{GPU_SERVER_IP}:{DIFF_PORT}")
    print(f"Connected to Diffusion Policy at {GPU_SERVER_IP}:{DIFF_PORT}")

    # Dino WM Socket
    dino_socket = context.socket(zmq.REQ)
    dino_socket.connect(f"tcp://{GPU_SERVER_IP}:{DINO_PORT}")
    print(f"Connected to Dino WM at {GPU_SERVER_IP}:{DINO_PORT}")


    
    robot = sim.SimRobotInterface()
    gripper = sim.SimGripperInterface()
    cameras = sim.SimDualCamera(H=IMG_H, W=IMG_W, hz=30)

    # 3. Buffers
    obs_buffer = deque(maxlen=NUM_HIST)
    action_buffer = deque(maxlen=NUM_HIST-1)

    # Pre-fill buffers with initial state and dummy actions
    print("Pre-filling buffers...")
    for _ in range(NUM_HIST):
        c1, c2 = cameras.get_frames() # Returns (C,H,W)
        s = robot.get_state()         # Returns (D,)
        obs_buffer.append({'c1': c1, 'c2': c2, 's': s})
        action_buffer.append(np.zeros(4, dtype=np.float32)) # Assuming 4D action [x,y,z,grip]
        time.sleep(0.1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    step_count = 0
    ep_count = 0

    try:
        while True:
            print(f"step: {step_count}")
            if step_count == 35:
                robot.reset()
                step_count = 0
                ep_count+=1
                obs_buffer.clear()
                action_buffer.clear()

            while len(obs_buffer) < 2:
                i1, i2 = cameras.get_frames()
                s = robot.get_state()
                obs_buffer.append({'c1': i1, 'c2': i2, 's': s})
                action_buffer.append(np.zeros(4, dtype=np.float32))

            # A. Prepare Data for Diffusion Policy
            # Stack history: (T, C, H, W)
            b_c1 = np.stack([x['c1'] for x in obs_buffer])
            b_c2 = np.stack([x['c2'] for x in obs_buffer])
            b_s = np.stack([x['s'] for x in obs_buffer])

            # B. Query Diffusion Policy
            diff_socket.send_pyobj({
                'cam1': b_c1.tobytes(),
                'cam2': b_c2.tobytes(),
                'state': b_s.tobytes(),
                'shape': (IMG_H, IMG_W)
            })
            
            diff_response = diff_socket.recv_pyobj()
            pred_actions = diff_response['action']

            # Dev Agent
            deviator = DeviatorAgent(pred_actions, b=PLAN_BATCH_SIZE)
            batch_pred_actions = deviator.perturbe() # Shape: (b, N, 4)
            
            # 1. Prepare Visual: Stack Cam1 and Cam2 on dim 1 -> (T, 2, C, H, W)
            vis_hist = np.stack([b_c1, b_c2], axis=1) 
            vis_hist = vis_hist[np.newaxis, ...] # Add Batch -> (1, T, 2, C, H, W)
            vis_hist = np.tile(vis_hist, (PLAN_BATCH_SIZE, 1, 1, 1, 1, 1)) # (b, T, 2, C, H, W)
            
            # 2. Prepare Proprio
            prop_hist = b_s[np.newaxis, ...] # (1, T, D)
            prop_hist = np.tile(prop_hist, (PLAN_BATCH_SIZE, 1, 1)) # (b, T, 4)

            # 3. Prepare Full Action Sequence
            hist_actions = np.stack(list(action_buffer)) # (T, D)
            batch_hist_actions = np.tile(hist_actions, (PLAN_BATCH_SIZE, 1, 1)) # (b, T_hist, 4)

            full_actions = np.concatenate([batch_hist_actions, batch_pred_actions], axis=1) # (b, T_hist + N, 4)

            dino_socket.send_pyobj({
                'visual': vis_hist,
                'proprio': prop_hist,
                'actions': full_actions
            })

            dino_response = dino_socket.recv_pyobj()

            # Predicted States: (1, N, 2, 3, H, W) usually
            if 'error' in dino_response:
                print(f"Dino Error: {dino_response['error']}")
                pred_imgs = None
                lyap_exps = None
                max_patch_indices = None
            else:
                pred_imgs = dino_response['states'] 
                lyap_exps = dino_response.get('lyapunov', None)
                
                # Retrieve the patch indices sent from the modified server.py
                max_patch_indices = dino_response.get('max_patch_idx', None) 

            if lyap_exps is not None:
                print(f"LP Mean: {np.mean(lyap_exps):.4f} | Max: {np.max(lyap_exps):.4f}")

            # D. Execute Actions & Capture Ground Truth
            gt_imgs_c1 = []
            gt_imgs_c2 = []
            
            for i, action in enumerate(pred_actions):
                # Execute
                robot.execute(action)
                
                # Step physics roughly to match control rate
                time.sleep(1.0 / CONTROL_HZ)
                c1, c2 = cameras.get_frames() # (C,H,W)
                    
                # Convert for saving/display (C,H,W) -> (H,W,C)
                gt_imgs_c1.append(c1.transpose(1, 2, 0))
                gt_imgs_c2.append(c2.transpose(1, 2, 0))

                # Capture GT Frame
                if i >= pred_actions.shape[0]-2:
                    # Update buffers for NEXT closed-loop iteration
                    s = robot.get_state()
                    obs_buffer.append({'c1': c1, 'c2': c2, 's': s})
                    action_buffer.append(action)

            if pred_imgs is not None:
                b_size, N_steps, _, _, _, _ = pred_imgs.shape
                display_imgs = []
                num_steps = min(N_steps, len(gt_imgs_c1))
                
                def add_label(img, text, color=(0, 255, 0)):
                    img_labeled = img.copy()
                    cv2.putText(img_labeled, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 1, cv2.LINE_AA)
                    return img_labeled

                for t in range(num_steps):
                    # 1. Ground Truth
                    gt_wrist = process_gt_image(gt_imgs_c1[t], 224)
                    gt_fixed = process_gt_image(gt_imgs_c2[t], 224)
                    
                    gt_wrist = add_label(gt_wrist, "GT Wrist", (255, 255, 255))
                    gt_fixed = add_label(gt_fixed, "GT Fixed", (255, 255, 255))
                    
                    wrist_column = [gt_wrist]
                    fixed_column = [gt_fixed]
                    
                    # 2. Iterate Batch
                    for batch_idx in range(b_size):
                        # Force ascontiguousarray so OpenCV can modify the image arrays without stride errors
                        p_wrist = np.ascontiguousarray(pred_imgs[batch_idx, t, 0].transpose(1, 2, 0))
                        p_fixed = np.ascontiguousarray(pred_imgs[batch_idx, t, 1].transpose(1, 2, 0))
                        
                        if batch_idx == 0:
                            label = "Orig Pred"
                            color = (0, 255, 0) # Green for original
                        else:
                            # Safely extract Lyapunov exponent if it exists
                            if lyap_exps is not None:
                                le_val = lyap_exps[batch_idx - 1]
                                if le_val > 0.5:
                                    color = (255, 0, 0)
                                else:
                                    color = (0, 0, 255)
                                label = f"Noisy {batch_idx} (LE: {le_val:.3f})"
                                color = (0, 0, 255) if le_val > 0 else (255, 0, 0) 
                                
                                # --- HIGHLIGHT THE MOST DIVERGENT PATCH ---
                                if max_patch_indices is not None:
                                    # batch_idx - 1 because max_patch_indices is for the noisy trajectories only
                                    patch_idx = max_patch_indices[batch_idx - 1]
                                    
                                    # Convert 1D index (0-195) to 14x14 grid coordinates
                                    row = patch_idx // 14
                                    col = patch_idx % 14
                                    
                                    # Scale up to 224x224 pixel space (16px per patch)
                                    x_start = col * 16
                                    y_start = row * 16
                                    
                                    # Draw a solid Red box (RGB: 255, 0, 0) directly on the fixed camera image
                                    cv2.rectangle(p_fixed, (x_start, y_start), 
                                                  (x_start + 16, y_start + 16), 
                                                  (255, 0, 0), thickness=2)
                                    
                            else:
                                label = f"Noisy Pred {batch_idx}"
                                color = (0, 255, 255) # Yellow fallback
                                
                        p_wrist = add_label(p_wrist, label, color)
                        p_fixed = add_label(p_fixed, label, color)
                        
                        wrist_column.append(p_wrist)
                        fixed_column.append(p_fixed)
                    
                    # 3. Stack
                    col_wrist_stacked = np.vstack(wrist_column)
                    col_fixed_stacked = np.vstack(fixed_column)
                    step_img = np.vstack([col_wrist_stacked, col_fixed_stacked])
                    display_imgs.append(step_img)

                # Horizontally stack all time steps left-to-right
                full_viz = np.hstack(display_imgs)
                
                # Convert RGB to BGR for OpenCV saving
                full_viz_bgr = cv2.cvtColor(full_viz, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(OUTPUT_DIR, f"ep_{ep_count}_step_{step_count}.png")
                cv2.imwrite(save_path, full_viz_bgr)

            step_count += 1

    except KeyboardInterrupt:
        print("Stopping...")
        if hasattr(robot, 'robot'):
            robot.robot.terminate_current_policy()

if __name__ == "__main__":
    main()