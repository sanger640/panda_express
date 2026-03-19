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
STRIDE = 1 
NUM_TEST_SAMPLES = 14  # Number of different trajectories to test

OUTPUT_DIR = "gt_multi_test_results"
# ---------------------

def process_image(img_bytes):
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.transpose(img, (2, 0, 1)) 

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    context = zmq.Context()
    dino_socket = context.socket(zmq.REQ)
    dino_socket.connect(f"tcp://{GPU_SERVER_IP}:{DINO_PORT}")
    print(f"🔗 Connected to Dino WM at {GPU_SERVER_IP}:{DINO_PORT}")

    print(f"📂 Opening LMDB at {LMDB_PATH}...")
    env = lmdb.open(LMDB_PATH, readonly=True, lock=False)
    
    with env.begin() as txn:
        meta_bytes = txn.get(b"__metadata__")
        metadata = pickle.loads(meta_bytes)
        ep_names = list(metadata["episodes"].keys())

        for sample_idx in range(NUM_TEST_SAMPLES):
            test_ep = random.choice(ep_names)
            ep_meta = metadata["episodes"][test_ep]
            
            # Ensure episode is long enough for the stride and prediction window
            max_start = ep_meta["seq_len"] - (NUM_HIST + NUM_PRED) - 1
            if max_start <= 0: continue
            
            start_step = random.randint(0, max_start)
            print(f"🧪 [{sample_idx+1}/{NUM_TEST_SAMPLES}] Episode: {test_ep} | Start Step: {start_step}")
            
            cam1_keys = ep_meta["keys"].get("cam1", [])
            cam2_keys = ep_meta["keys"].get("cam2", [])
            
            act_bytes = txn.get(f"{test_ep}_actions".encode('ascii'))
            proc_bytes = txn.get(f"{test_ep}_proprio".encode('ascii'))
            actions = pickle.loads(act_bytes) 
            proprios = pickle.loads(proc_bytes)

            total_steps = NUM_HIST + NUM_PRED
            vis_c1, vis_c2, gt_visuals = [], [], []
            
            for step_offset in range(total_steps):
                current_step = start_step + step_offset
                img_idx = current_step * STRIDE 
                
                img_c1 = process_image(txn.get(cam1_keys[img_idx].encode('ascii')))
                img_c2 = process_image(txn.get(cam2_keys[img_idx].encode('ascii')))
                
                vis_c1.append(img_c1)
                vis_c2.append(img_c2)
                gt_visuals.append((
                    cv2.cvtColor(np.transpose(img_c1, (1, 2, 0)), cv2.COLOR_RGB2BGR),
                    cv2.cvtColor(np.transpose(img_c2, (1, 2, 0)), cv2.COLOR_RGB2BGR)
                ))

            # Prepare Tensors
            b_c1 = np.stack(vis_c1[:NUM_HIST])
            b_c2 = np.stack(vis_c2[:NUM_HIST])
            vis_hist = np.stack([b_c1, b_c2], axis=1)[np.newaxis, ...]
            prop_hist = proprios[start_step : start_step + NUM_HIST][np.newaxis, ...]
            plan_actions = actions[start_step : start_step + total_steps][np.newaxis, ...]

            # Query Server
            dino_socket.send_pyobj({
                'visual': vis_hist.astype(np.uint8),
                'proprio': prop_hist.astype(np.float32),
                'actions': plan_actions.astype(np.float32)
            })
            
            response = dino_socket.recv_pyobj()
            if 'error' in response:
                print(f"❌ Error on {test_ep}: {response['error']}")
                continue
                
            p_seq = response['states'][0] 
            p_future = p_seq[NUM_HIST:] # Aligns with future steps
            
            display_rows = []
            for t in range(NUM_PRED):
                gt_idx = NUM_HIST + t
                gt_c1, gt_c2 = gt_visuals[gt_idx]
                gt_c1, gt_c2 = cv2.resize(gt_c1, (224, 224)), cv2.resize(gt_c2, (224, 224))
                
                p_c1 = cv2.cvtColor(np.ascontiguousarray(p_future[t, 0].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                p_c2 = cv2.cvtColor(np.ascontiguousarray(p_future[t, 1].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR)
                
                # Labels
                for img, txt, col in [(gt_c1, "GT C1", (0,255,0)), (p_c1, "Pred C1", (0,255,255)), 
                                      (gt_c2, "GT C2", (0,255,0)), (p_c2, "Pred C2", (0,255,255))]:
                    cv2.putText(img, f"{txt} t+{t+1}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

                display_rows.append(np.hstack([np.vstack([gt_c1, gt_c2]), np.vstack([p_c1, p_c2])]))
            
            final_img = np.hstack(display_rows)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"sample_{sample_idx}_{test_ep}_step{start_step}.png"), final_img)

    env.close()
    print(f"✅ All test samples saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()