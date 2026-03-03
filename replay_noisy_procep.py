import time
import os
import json
import random
import glob
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import traceback
import shutil

# Import interfaces directly from sim.py
from sim import SimRobotInterface, SimGripperInterface, SimRecorder, SIM

# --- CONFIGURATION ---
SOURCE_TASK_DIR = "tasks/jenga_mujoco/"
TARGET_TASK_DIR = "tasks/jenga_mujoco_noise/"  # New folder for noisy data

# How many new noisy episodes to generate
N_EPISODES_TO_GENERATE = 1000

# Dummy serials (required by SimRecorder class but ignored in Sim)
CAM1_SERIAL = "215222078938"
CAM2_SERIAL = "819612070440"

# --- NOISE PARAMETERS (Tuned for Jenga) ---
# Fixed noise is safer than relative noise for delicate tasks.
# 2mm position noise is enough to robustify without knocking over the tower.
NOISE_POS_STD = 0.007    # +/- 2mm standard deviation
NOISE_ROT_STD = 0.05    # +/- ~1 degree (0.017 rad) standard deviation

def get_saved_trajectories(root_dir):
    """Recursively finds all trajectory_*.json files in the source directory."""
    search_path = os.path.join(root_dir, "episodes", "*", "trajectory_*.json")
    files = glob.glob(search_path)
    if not files:
        print(f"[WARN] No trajectories found in {search_path}")
    return files

def get_next_episode_id(target_root):
    """Finds the next available episode ID in the target folder."""
    episodes_dir = os.path.join(target_root, "episodes")
    if not os.path.exists(episodes_dir):
        return 0
    
    existing_ids = [int(d) for d in os.listdir(episodes_dir) if d.isdigit()]
    if not existing_ids:
        return 0
    return max(existing_ids) + 1

def apply_fixed_noise(pos, quat, gripper):
    """
    Applies fixed Gaussian noise to the target pose.
    """
    # 1. Position Noise (Add 2mm jitter)
    noise_pos = np.random.normal(0, NOISE_POS_STD, 3)
    noisy_pos = np.array(pos) + noise_pos

    # 2. Rotation Noise (Add 1 deg jitter)
    # current_rot = R.from_quat(quat)
    # noise_rot_vec = np.random.normal(0, NOISE_ROT_STD, 3)
    # noise_rot = R.from_rotvec(noise_rot_vec)
    # noisy_rot = current_rot * noise_rot
    # noisy_quat = noisy_rot.as_quat()

    # 3. Gripper (Pass-through)
    # Adding noise to the gripper in Jenga usually just causes failure (dropping block).
    # We keep the original gripper intent.
    return noisy_pos, quat, gripper

def main():
    print(f"[REPLAY] Starting Noise Augmentation...")
    print(f"[CONFIG] Source: {SOURCE_TASK_DIR}")
    print(f"[CONFIG] Target: {TARGET_TASK_DIR}")
    
    # 1. Initialize Interfaces
    robot = SimRobotInterface()
    gripper = SimGripperInterface()
    
    # 2. Find Source Data
    traj_files = get_saved_trajectories(SOURCE_TASK_DIR)
    if not traj_files: return
    print(f"[REPLAY] Found {len(traj_files)} source trajectories.")

    # 3. Determine Start Index for New Data
    current_idx = get_next_episode_id(SOURCE_TASK_DIR)
    recorder = SimRecorder(current_idx, CAM1_SERIAL, CAM2_SERIAL, TARGET_TASK_DIR)
    print(f"[REPLAY] Writing new episodes starting at ID: {current_idx}")

    for n in range(N_EPISODES_TO_GENERATE):
        try:
            # --- PHASE 1: PRE-CALCULATION ---
            
            # A. Select Random Source Episode
            source_file = random.choice(traj_files)
            print(f"\n--- Generating Episode {current_idx} (Source: {os.path.basename(source_file)}) ---")
            
            with open(source_file, 'r') as f:
                data = json.load(f)
            source_waypoints = data['waypoints']
            if not source_waypoints: continue

            # B. Generate the "Noisy Plan"
            # We calculate targets first so we can iterate deterministically
            noisy_plan = []
            for wp in source_waypoints:
                n_pos, n_quat, n_grip = apply_fixed_noise(
                    wp['position'], 
                    wp['orientation'], 
                    wp['gripper']
                )
                noisy_plan.append({
                    'target_pos': n_pos,
                    'target_quat': n_quat,
                    'target_grip': n_grip,
                    'original_timestamp': wp['timestamp']
                })

            # --- PHASE 2: EXECUTION ---

            # C. Reset Simulation
            robot.reset()
            time.sleep(0.5) # Let physics settle

            # D. Start Recording (To TARGET_TASK_DIR)
            # SimRecorder will automatically create the folder structure
            recorder.save_folder = TARGET_TASK_DIR+"episodes/" + str(current_idx) + "/rgb_frames/"
            os.makedirs(recorder.save_folder, exist_ok=True)
            recorder.start()
            
            # E. Execute & Record Ground Truth
            actual_trajectory_data = []
            
            start_real_time = time.time()
            start_sim_time = noisy_plan[0]['original_timestamp']
            
            for step in noisy_plan:
                # 1. Timing Control (Playback Speed)
                target_time = step['original_timestamp'] - start_sim_time
                elapsed = time.time() - start_real_time
                
                if target_time > elapsed:
                    time.sleep(target_time - elapsed)

                # Process intended targets (Intent)
                target_pos_tensor = torch.Tensor(step['target_pos'])
                target_quat_tensor = torch.Tensor(step['target_quat'])

                grip_val = step['target_grip']
                if isinstance(grip_val, bool):
                    is_closed = grip_val
                else:
                    is_closed = grip_val > 0.5

                # 2. Observe ACTUAL State (Physical Result BEFORE execution)
                curr_pos_tensor, curr_quat_tensor = robot.get_ee_pose()
                actual_grip_bool = SIM.gripper_val < 10.0

                # 3. Record the pairing (State + Intent)
                actual_trajectory_data.append({
                    'timestamp': time.time(),
                    'position': target_pos_tensor.numpy().tolist(),
                    'orientation': target_quat_tensor.numpy().tolist(),
                    'gripper': is_closed,
                    'proc_pos': curr_pos_tensor.numpy().tolist(),
                    'proc_quat': curr_quat_tensor.numpy().tolist(),
                    'proc_gripper': actual_grip_bool
                })

                # 4. Execute Command (Physical movement towards future)
                robot.update_desired_ee_pose(
                    target_pos_tensor, 
                    target_quat_tensor
                )

                if is_closed:
                    gripper.grasp(0.1, 0.1) # Close
                else:
                    gripper.goto(255, 0.1, 0.1) # Open

                # Preview
                recorder.show_preview(is_recording=True)

            # F. Finish Episode
            recorder.stop()
            
            # --- PHASE 3: SAVE ---
            # Save trajectory JSON to the new folder
            dest_folder = os.path.join(TARGET_TASK_DIR, "episodes", str(current_idx))
            os.makedirs(dest_folder, exist_ok=True)
            output_json = os.path.join(dest_folder, f"trajectory_{int(time.time())}.json")
            
            save_data = {
                'metadata': {
                    'source_episode': source_file,
                    'num_waypoints': len(actual_trajectory_data),
                    'noise_injected': True,
                    'noise_params': {'pos_std': NOISE_POS_STD, 'rot_std': NOISE_ROT_STD}
                },
                'waypoints': actual_trajectory_data
            }
            
            with open(output_json, 'w') as f:
                json.dump(save_data, f, indent=2)
                
            print(f"[SAVE] Saved new episode {current_idx} to {TARGET_TASK_DIR}")
            current_idx += 1

        except Exception as e:
            print(f"[ERROR] Failed episode generation: {e}")
            traceback.print_exc()
            try: recorder.stop()
            except: pass

    print("[DONE] Noise Augmentation Complete.")

if __name__ == "__main__":
    main()