import os
import zmq
import lmdb
import pickle
import numpy as np
import cv2
import time

# --- CONFIGURATION ---
GPU_SERVER_IP = "localhost" 
DINO_PORT = 5556
LMDB_PATH = "/app/tasks/jenga_mujoco_noise/jenga_single.lmdb" 

NUM_HIST = 3
NUM_PRED = 8
PLAN_BATCH_SIZE = 2 # 1 Original + 31 Noisy trajectories
NOISE_STD = 5e-3
LE_T = 0.87 # Threshold for High LE alerts
OUTPUT_DIR = "test_pca"
# ---------------------

class DeviatorAgent():
    def __init__(self, act_seq, b, std_dev=5e-4):
        self.act_seq = act_seq # T x 4
        self.b = b
        self.std_dev = std_dev
        
    def perturb(self):
        A = np.tile(self.act_seq, (self.b, 1, 1))
        if self.b > 1:
            T = self.act_seq.shape[1]
            noise = np.random.normal(0, self.std_dev, size=(self.b - 1, T, 3))
            A[1:, :, :3] += noise
        return A.astype(np.float32)

def process_image(img_bytes):
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.transpose(img, (2, 0, 1)) 

def center_crop_resize(img, size=224):
    """Replicates PyTorch's Resize(size) + CenterCrop(size) to prevent squashing."""
    h, w = img.shape[:2]
    
    if h < w:
        new_h = size
        new_w = int(w * (size / h))
    else:
        new_w = size
        new_h = int(h * (size / w))
        
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    start_y = (new_h - size) // 2
    start_x = (new_w - size) // 2
    cropped = resized[start_y : start_y + size, start_x : start_x + size]
    
    return cropped

def add_label(img, text, color=(0, 255, 0), bg_color=(0, 0, 0), alpha=0.6):
    """Draws text over a semi-transparent background box for high readability."""
    res = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    thickness = 1
    
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = 5, 15
    pad = 3
    top_left = (max(0, x - pad), max(0, y - text_height - pad))
    bottom_right = (x + text_width + pad, y + baseline + pad - 2)
    
    overlay = res.copy()
    cv2.rectangle(overlay, top_left, bottom_right, bg_color, -1)
    
    cv2.addWeighted(overlay, alpha, res, 1 - alpha, 0, res)
    cv2.putText(res, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    
    return res

def overlay_pca(base_img, pca_img):
    """Blends the PCA semantic mask onto the decoded image."""
    gray_pca = cv2.cvtColor(pca_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_pca, 1, 255, cv2.THRESH_BINARY)
    
    blended = base_img.copy()
    foreground = cv2.addWeighted(base_img, 0.4, pca_img, 0.8, 0)
    
    np.copyto(blended, foreground, where=(mask[:,:,None] == 255))
    return blended

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{GPU_SERVER_IP}:{DINO_PORT}")
    
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    with env.begin() as txn:
        metadata = pickle.loads(txn.get(b"__metadata__"))
        ep_names = list(metadata["episodes"].keys())

        print(f"Found {len(ep_names)} total episodes. Beginning contiguous window evaluation...")

        # --- LOOP 1: Iterate through EVERY episode ---
        for ep_idx, test_ep in enumerate(ep_names):
            ep_meta = metadata["episodes"][test_ep]
            c2_keys = ep_meta["keys"]["cam2"]
            
            act_all = pickle.loads(txn.get(f"{test_ep}_actions".encode('ascii')))
            prop_all = pickle.loads(txn.get(f"{test_ep}_proprio".encode('ascii')))
            
            actual_seq_len = min(len(c2_keys), len(act_all), len(prop_all))
            max_start = actual_seq_len - (NUM_HIST + NUM_PRED) - 1
            
            if max_start <= 0: 
                print(f"Skipping {test_ep} (Actual length {actual_seq_len} is too short)")
                continue
                
            ep_output_dir = os.path.join(OUTPUT_DIR, test_ep)
            os.makedirs(ep_output_dir, exist_ok=True)
            
            num_chunks = (max_start // NUM_PRED) + 1
            print(f"\n--- Processing Episode {ep_idx+1}/{len(ep_names)}: {test_ep} ({num_chunks} discrete windows) ---")

            # --- LOOP 2: Jump the window across the episode ---
            for start_step in range(0, max_start + 1, NUM_PRED):
                skip_episode = False
                
                vis_c2, gt_visuals = [], []
                for t_off in range(NUM_HIST + NUM_PRED):
                    idx = start_step + t_off
                    
                    if idx >= len(c2_keys):
                        print(f"  ⚠️ Index {idx} out of bounds. Reached end of {test_ep} prematurely.")
                        skip_episode = True
                        break
                        
                    img_bytes = txn.get(c2_keys[idx].encode('ascii'))
                    
                    if img_bytes is None:
                        print(f"  ⚠️ Missing image at index {idx} in {test_ep}. Skipping to next episode.")
                        skip_episode = True
                        break
                        
                    img2 = process_image(img_bytes)
                    vis_c2.append(img2)
                    gt_visuals.append(cv2.cvtColor(np.transpose(img2, (1,2,0)), cv2.COLOR_RGB2BGR))

                if skip_episode:
                    break 

                clean_actions = act_all[start_step : start_step + NUM_HIST + NUM_PRED][np.newaxis, ...]
                deviator = DeviatorAgent(clean_actions, b=PLAN_BATCH_SIZE, std_dev=NOISE_STD)
                batch_actions = deviator.perturb()
                
                b_c2 = np.stack(vis_c2[:NUM_HIST])
                vis_hist = np.tile(b_c2[np.newaxis, ...], (PLAN_BATCH_SIZE, 1, 1, 1, 1))
                prop_hist = np.tile(prop_all[start_step : start_step + NUM_HIST][np.newaxis, ...], (PLAN_BATCH_SIZE, 1, 1))

                socket.send_pyobj({'visual': vis_hist.astype(np.uint8), 'proprio': prop_hist.astype(np.float32), 'actions': batch_actions})
                resp = socket.recv_pyobj()
                
                if 'error' in resp: 
                    print(f"Server Error at step {start_step}: {resp['error']}")
                    continue

                pred_imgs = resp['states'] 
                pca_imgs = resp.get('pca_mask')
                max_le_val = resp.get('max_lyapunov', 0.0)
                max_patch_idx = resp.get('max_patch_idx', 0)
                lyaps = resp.get('all_lyapunovs', np.array([]))
                
                num_triggers = np.sum(lyaps > LE_T) if len(lyaps) > 0 else 0
                
                # --- HIGH LE CHECK & VISUALIZATION ASSEMBLY ---
                if max_le_val > LE_T:
                    print(f"  🚨 HIGH LE DETECTED | Step: {start_step:03d} | Max LE: {max_le_val:.2f} | Triggers: {num_triggers}/{PLAN_BATCH_SIZE-1}")
                
                time_cols = []
                for t in range(NUM_PRED):
                    gt_idx = NUM_HIST + t
                    
                    # Row 1: Ground Truth
                    gt_frame = center_crop_resize(gt_visuals[gt_idx], size=224)
                    col_frames = [add_label(gt_frame, "GROUND TRUTH", (0, 255, 0))]
                    
                    # Row 2: Original Prediction (Decoded)
                    orig_p = cv2.cvtColor(np.ascontiguousarray(pred_imgs[0, gt_idx].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                    col_frames.append(add_label(orig_p, "ORIG PRED", (255, 255, 255)))

                    # Row 3: Original Prediction (PCA Overlay)
                    orig_pca = cv2.cvtColor(np.ascontiguousarray(pca_imgs[0, gt_idx].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                    orig_blended = overlay_pca(orig_p, orig_pca)
                    col_frames.append(orig_blended)
                    
                    # Row 4 & 5: Worst Case Noisy Prediction
                    if pred_imgs.shape[0] > 1:
                        worst_p = cv2.cvtColor(np.ascontiguousarray(pred_imgs[1, gt_idx].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                        worst_pca = cv2.cvtColor(np.ascontiguousarray(pca_imgs[1, gt_idx].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                        worst_blended = overlay_pca(worst_p, worst_pca)
                        
                        # Highlight the maximum LE patch on both Worst representations
                        if t == NUM_PRED - 1:
                            p_idx = max_patch_idx
                            y, x = (p_idx // 14) * 16, (p_idx % 14) * 16
                            # rect_color = (0, 0, 255) # Red because it breached LE_T
                            # cv2.rectangle(orig_blended, (x, y), (x+16, y+16), rect_color, 2)
                            # cv2.rectangle(worst_p, (x, y), (x+16, y+16), rect_color, 2)
                            # cv2.rectangle(worst_blended, (x, y), (x+16, y+16), rect_color, 2)
                            
                        col_frames.append(add_label(worst_p, f"WORST NOISY PRED (LE: {max_le_val:.2f})", (0, 0, 255)))
                        col_frames.append(add_label(worst_blended, f"WORST MASK [Triggers: {num_triggers}]", (0, 0, 255)))

                    time_cols.append(np.vstack(col_frames))

                final_viz = np.hstack(time_cols)
                save_name = f"step_{start_step:04d}.png" 
                cv2.imwrite(os.path.join(ep_output_dir, save_name), final_viz)
                print(f"  Saved {save_name}")

    env.close()
    print("\nExhaustive evaluation complete.")

if __name__ == "__main__":
    main()