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
# ---------------------

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

    # diff_socket.setsockopt(zmq.RCVTIMEO, 5000) # 5 second timeout
    # dino_socket.setsockopt(zmq.RCVTIMEO, 5000)

    # 2. Setup Simulation
    # Initialize the MuJoCo physics context (starts a background thread)
    # sim_context = SimContext()
    # Give it a moment to stabilize
    # time.sleep(1.0)
    
    # Interfaces (Using the Sim versions)
    # Note: sim_context is a global singleton in muj_franka_test, 
    # but we instantiated it to ensure physics runs.
    # The interfaces usually access the global instance or we pass it if modified.
    # Based on your provided file, SimRobotInterface accesses global 'SIM', 
    # so we need to inject our instance into the module's global scope if needed, 
    # or rely on the fact that SimContext sets itself up.


    
    robot = sim.SimRobotInterface()
    gripper = sim.SimGripperInterface()
    cameras = sim.SimDualCamera(H=IMG_H, W=IMG_W, hz=30)

    # 3. Buffers
    obs_buffer = deque(maxlen=NUM_HIST)
    action_buffer = deque(maxlen=NUM_HIST)

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

    try:
        while True:
            print(f"\n--- Step {step_count} ---")
            
            # A. Prepare Data for Diffusion Policy
            # Stack history: (T, C, H, W)
            b_c1 = np.stack([x['c1'] for x in obs_buffer])
            b_c2 = np.stack([x['c2'] for x in obs_buffer])
            b_s = np.stack([x['s'] for x in obs_buffer])

            # B. Query Diffusion Policy
            # print("Querying Diffusion Policy...")
            diff_socket.send_pyobj({
                'cam1': b_c1.tobytes(),
                'cam2': b_c2.tobytes(),
                'state': b_s.tobytes(),
                'shape': (IMG_H, IMG_W)
            })
            
            diff_response = diff_socket.recv_pyobj()
            print("umm diffy works!")
            # actions shape: (N_pred, Action_Dim) e.g., (8, 4)
            pred_actions = diff_response['action'] 
            
            # C. Query Dino World Model
            # Dino needs: 
            # 1. Visual History (1, T, 2, C, H, W) - stacked views
            # 2. Proprio History (1, T, D)
            # 3. Action Sequence (1, T + N_pred, D) - History + Future
            
            print("Querying Dino World Model...")
            
            # 1. Prepare Visual: Stack Cam1 and Cam2 on dim 1 -> (T, 2, C, H, W)
            # Ensure range 0-255 uint8 if that's what cameras return
            vis_hist = np.stack([b_c1, b_c2], axis=1) 
            vis_hist = vis_hist[np.newaxis, ...] # Add Batch -> (1, T, 2, C, H, W)

            # 2. Prepare Proprio
            prop_hist = b_s[np.newaxis, ...] # (1, T, D)

            # 3. Prepare Full Action Sequence
            hist_actions = np.stack(list(action_buffer)) # (T, D)
            full_actions = np.concatenate([hist_actions, pred_actions], axis=0) # (T+N, D)
            full_actions = full_actions[np.newaxis, ...] # (1, T+N, D)
            print("vis_history")
            print(vis_hist.shape)
            print("full actions")
            print(full_actions.shape)

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
            else:
                pred_imgs = dino_response['states'] 
                # Shape check: (1, N, 2, 3, H, W)
                # We typically want to visualize the future predictions corresponding to pred_actions

            # D. Execute Actions & Capture Ground Truth
            gt_imgs_c1 = []
            gt_imgs_c2 = []
            
            print(f"Executing {len(pred_actions)} actions...")
            for i, action in enumerate(pred_actions):
                # Execute
                robot.execute(action)
                
                # Step physics roughly to match control rate
                # SimContext runs its own thread, but execute updates the target.
                # We sleep to let physics settle for 1/Hz
                time.sleep(1.0 / CONTROL_HZ)
                
                # Capture GT Frame
                c1, c2 = cameras.get_frames() # (C,H,W)
                
                # Convert for saving/display (C,H,W) -> (H,W,C)
                gt_imgs_c1.append(c1.transpose(1, 2, 0))
                gt_imgs_c2.append(c2.transpose(1, 2, 0))

                # Update buffers for NEXT closed-loop iteration
                # (Even though we are running open-loop for this sequence, 
                # we update buffers so the next major step is continuous)
                s = robot.get_state()
                obs_buffer.append({'c1': c1, 'c2': c2, 's': s})
                action_buffer.append(action)

            # E. Visualize / Save Results
            if pred_imgs is not None:
                # Process Predictions
                # pred_imgs shape: (1, N, 2, 3, H, W)
                # Extract first batch
                preds = pred_imgs[0] # (N, 2, 3, H, W)
                
                display_imgs = []
                
                # Loop through predicted steps
                num_steps = min(len(preds), len(gt_imgs_c1))
                
                for t in range(num_steps):
                    # Ground Truth (Cam1) - Top Row
                    gt_img = gt_imgs_c1[t] # Already HWC, RGB
                    
                    # Prediction (Cam1 is index 0 or 1 depending on model training)
                    # Assuming Index 1 is Front (Cam1) and Index 0 is Wrist based on standard DinoWM config?
                    # Or check SimDualCamera: Cam1 (Wrist?) vs Cam2 (Fixed?)
                    # Let's verify SimDualCamera:
                    #   renderer.update_scene(..., "cam_wrist") -> img1
                    #   renderer.update_scene(..., "cam_fixed") -> img2
                    # So Cam1 = Wrist, Cam2 = Fixed.
                    # DinoWM usually treats View 0 as Wrist, View 1 as Front.
                    
                    # Let's stack both views for a complete comparison
                    # GT Column: [Wrist GT]
                    #            [Fixed GT]
                    gt_wrist = gt_imgs_c1[t]
                    gt_fixed = gt_imgs_c2[t]
                    
                    # Pred Column
                    # preds[t, 0] -> Wrist Pred (3, H, W) -> Transpose to HWC
                    # preds[t, 1] -> Fixed Pred
                    pred_wrist = preds[t, 0].transpose(1, 2, 0)
                    pred_fixed = preds[t, 1].transpose(1, 2, 0)
                    
                    # Stitch: Top=GT, Bottom=Pred
                    # Col: Wrist | Fixed
                    
                    # Create Column for this timestep
                    #  [ GT Wrist ]
                    #  [ Pr Wrist ]
                    col_wrist = np.vstack([gt_wrist, pred_wrist])
                    
                    #  [ GT Fixed ]
                    #  [ Pr Fixed ]
                    col_fixed = np.vstack([gt_fixed, pred_fixed])
                    
                    # Combined Step Image
                    step_img = np.vstack([col_wrist, col_fixed])
                    
                    display_imgs.append(step_img)

                # Horizontally stack time steps
                full_viz = np.hstack(display_imgs)
                
                # Save
                # Convert RGB to BGR for OpenCV saving
                full_viz_bgr = cv2.cvtColor(full_viz, cv2.COLOR_RGB2BGR)
                save_path = os.path.join(OUTPUT_DIR, f"step_{step_count}.png")
                cv2.imwrite(save_path, full_viz_bgr)
                print(f"Saved visualization to {save_path}")

            step_count += 1

    except KeyboardInterrupt:
        print("Stopping...")
        if hasattr(robot, 'robot'):
            robot.robot.terminate_current_policy()

if __name__ == "__main__":
    main()