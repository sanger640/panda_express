import os
import zmq
import lmdb
import pickle
import numpy as np
import cv2
import time
import random

# --- CONFIGURATION ---
GPU_SERVER_IP = "localhost" 
DINO_PORT = 5556
LMDB_PATH = "/home/sanger/jenga_mujoco_noise/jenga_unified.lmdb" 

NUM_HIST = 3
NUM_PRED = 8
PLAN_BATCH_SIZE = 3 # 1 Original + 2 Noisy trajectories
NOISE_STD = 5e-3
NUM_TEST_SAMPLES = 5

OUTPUT_DIR = "gt_deviator_results"
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

def add_label(img, text, color=(0, 255, 0)):
    res = img.copy()
    cv2.putText(res, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return res

def overlay_pca(base_img, pca_img):
    """Blends the PCA semantic mask onto the decoded image."""
    # Convert pure black background in PCA to a mask so it doesn't dim the base image
    gray_pca = cv2.cvtColor(pca_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_pca, 1, 255, cv2.THRESH_BINARY)
    
    # Blend only where the PCA mask exists (foreground)
    blended = base_img.copy()
    foreground = cv2.addWeighted(base_img, 0.4, pca_img, 0.8, 0)
    
    # Combine background from base and foreground from blended
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

        for s_idx in range(NUM_TEST_SAMPLES):
            test_ep = random.choice(ep_names)
            ep_meta = metadata["episodes"][test_ep]
            max_start = ep_meta["seq_len"] - (NUM_HIST + NUM_PRED) - 1
            if max_start <= 0: continue
            start_step = random.randint(0, max_start)
            
            # --- Gather GT Data ---
            act_all = pickle.loads(txn.get(f"{test_ep}_actions".encode('ascii')))
            prop_all = pickle.loads(txn.get(f"{test_ep}_proprio".encode('ascii')))
            
            vis_c1, vis_c2, gt_visuals = [], [], []
            for t_off in range(NUM_HIST + NUM_PRED):
                idx = start_step + t_off
                c1_keys, c2_keys = ep_meta["keys"]["cam1"], ep_meta["keys"]["cam2"]
                img1 = process_image(txn.get(c1_keys[idx].encode('ascii')))
                img2 = process_image(txn.get(c2_keys[idx].encode('ascii')))
                vis_c1.append(img1); vis_c2.append(img2)
                gt_visuals.append((cv2.cvtColor(np.transpose(img1, (1,2,0)), cv2.COLOR_RGB2BGR),
                                   cv2.cvtColor(np.transpose(img2, (1,2,0)), cv2.COLOR_RGB2BGR)))

            # --- Prepare Batch via Deviator ---
            clean_actions = act_all[start_step : start_step + NUM_HIST + NUM_PRED][np.newaxis, ...]
            deviator = DeviatorAgent(clean_actions, b=PLAN_BATCH_SIZE, std_dev=NOISE_STD)
            batch_actions = deviator.perturb()
            
            b_c1 = np.stack(vis_c1[:NUM_HIST]); b_c2 = np.stack(vis_c2[:NUM_HIST])
            vis_hist = np.tile(np.stack([b_c1, b_c2], axis=1)[np.newaxis, ...], (PLAN_BATCH_SIZE, 1, 1, 1, 1, 1))
            prop_hist = np.tile(prop_all[start_step : start_step + NUM_HIST][np.newaxis, ...], (PLAN_BATCH_SIZE, 1, 1))

            socket.send_pyobj({'visual': vis_hist.astype(np.uint8), 'proprio': prop_hist.astype(np.float32), 'actions': batch_actions})
            resp = socket.recv_pyobj()
            if 'error' in resp: continue

            # Extract both arrays
            pred_imgs = resp['states'] 
            pca_imgs = resp['pca_mask']
            
            lyaps = resp.get('lyapunov', None)
            patch_indices = resp.get('max_patch_idx', None)
            
            # --- Visualization Assembly ---
            time_cols = []
            for t in range(NUM_PRED):
                gt_idx = NUM_HIST + t
                gt_frame = cv2.resize(gt_visuals[gt_idx][1], (224, 224))
                col_frames = [add_label(gt_frame, "GROUND TRUTH", (0,255,0))]
                
                # 1. Original Prediction (Decoded)
                orig_p = cv2.cvtColor(np.ascontiguousarray(pred_imgs[0, gt_idx, 1].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                col_frames.append(add_label(orig_p, "ORIG PRED", (255,255,255)))
                
                # 2. Original Prediction (PCA Overlay)
                orig_pca = cv2.cvtColor(np.ascontiguousarray(pca_imgs[0, gt_idx, 1].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                orig_blended = overlay_pca(orig_p, orig_pca)
                col_frames.append(add_label(orig_blended, "ORIG PCA MASK", (255,100,255)))
                
                for b in range(1, PLAN_BATCH_SIZE):
                    # 3. Noisy Prediction (Decoded)
                    noisy_p = cv2.cvtColor(np.ascontiguousarray(pred_imgs[b, gt_idx, 1].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                    
                    # 4. Noisy Prediction (PCA Overlay)
                    noisy_pca = cv2.cvtColor(np.ascontiguousarray(pca_imgs[b, gt_idx, 1].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                    noisy_blended = overlay_pca(noisy_p, noisy_pca)
                    
                    le_val = lyaps[b-1] if lyaps is not None else 0.0
                    
                    # Highlight Patch based on LE threshold on the Blended image
                    if t == NUM_PRED - 1 and patch_indices is not None:
                        p_idx = patch_indices[b-1]
                        y, x = (p_idx // 14) * 16, (p_idx % 14) * 16
                        rect_color = (0, 255, 0) if le_val < 1.0 else (0, 0, 255)
                        # Draw on the blended image
                        cv2.rectangle(noisy_blended, (x, y), (x+16, y+16), rect_color, 2)
                    
                    col_frames.append(add_label(noisy_p, f"NOISY {b} PRED", (0,255,255)))
                    col_frames.append(add_label(noisy_blended, f"NOISY {b} MASK (LE:{le_val:.2f})", (0,255,255)))
                
                time_cols.append(np.vstack(col_frames))

            final_viz = np.hstack(time_cols)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"deviator_{s_idx}_{test_ep}.png"), final_viz)
            print(f"✅ Saved Deviator Test: {test_ep}")

    env.close()

if __name__ == "__main__":
    main()