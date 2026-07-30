import argparse
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
from sim import SimRobotInterface, SimGripperInterface, SimRecorder, SIM, TOPPLE_THRESHOLD_DEG

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
    global SOURCE_TASK_DIR, TARGET_TASK_DIR, N_EPISODES_TO_GENERATE
    global NOISE_POS_STD, NOISE_ROT_STD

    ap = argparse.ArgumentParser(description="Replay teleop episodes with injected noise.")
    ap.add_argument("--n-episodes", type=int, default=N_EPISODES_TO_GENERATE)
    ap.add_argument("--source-dir", default=SOURCE_TASK_DIR)
    ap.add_argument("--target-dir", default=TARGET_TASK_DIR)
    ap.add_argument("--noise-pos", type=float, default=NOISE_POS_STD,
                    help="Position noise std in metres. NOTE: the historical default 0.007 "
                         "is 7mm, though the comment beside it says 2mm (=0.002).")
    ap.add_argument("--noise-rot", type=float, default=NOISE_ROT_STD)
    args = ap.parse_args()
    SOURCE_TASK_DIR = args.source_dir
    TARGET_TASK_DIR = args.target_dir
    N_EPISODES_TO_GENERATE = args.n_episodes
    NOISE_POS_STD = args.noise_pos
    NOISE_ROT_STD = args.noise_rot

    print(f"[REPLAY] Starting Noise Augmentation...")
    print(f"[CONFIG] Source: {SOURCE_TASK_DIR}")
    print(f"[CONFIG] Target: {TARGET_TASK_DIR}")
    print(f"[CONFIG] Episodes: {N_EPISODES_TO_GENERATE}")
    print(f"[CONFIG] Noise: pos_std={NOISE_POS_STD} m ({NOISE_POS_STD*1000:.1f} mm), rot_std={NOISE_ROT_STD}")

    # 1. Initialize Interfaces
    robot = SimRobotInterface()
    gripper = SimGripperInterface()

    # 2. Find Source Data
    traj_files = get_saved_trajectories(SOURCE_TASK_DIR)
    if not traj_files: return
    print(f"[REPLAY] Found {len(traj_files)} source trajectories.")

    # 3. Determine Start Index for New Data
    # Index off the TARGET, not the source: re-running must append rather than
    # overwrite episodes already generated (an 8h job needs to be resumable).
    current_idx = get_next_episode_id(TARGET_TASK_DIR)
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
            # recorder.i = current_idx
            # os.path.join, not concatenation: the old form silently required
            # TARGET_TASK_DIR to end in "/" (the hardcoded default did, --target-dir did
            # not), sending every frame to "<dir>episodes/..." while the trajectory JSON
            # below used os.path.join and landed correctly -- so episodes looked complete
            # with zero images in them.
            recorder.save_folder = os.path.join(TARGET_TASK_DIR, "episodes",
                                                str(current_idx), "rgb_frames")
            os.makedirs(recorder.save_folder, exist_ok=True)
            # print("rec save foder")
            # print(recorder.save_folder)
            recorder.start()
            
            # E. Execute & Record Ground Truth
            actual_trajectory_data = []
            failure_step = None
            failure_timestamp = None
            failure_block = None
            peak_tilt, peak_block = 0.0, None
            
            start_real_time = time.time()
            start_sim_time = noisy_plan[0]['original_timestamp']
            
            for step in noisy_plan:
                # 1. Timing Control (Playback Speed)
                target_time = step['original_timestamp'] - start_sim_time
                elapsed = time.time() - start_real_time
                
                if target_time > elapsed:
                    time.sleep(target_time - elapsed)

                # 2. Send Command (Noisy Target)
                robot.update_desired_ee_pose(
                    torch.Tensor(step['target_pos']), 
                    torch.Tensor(step['target_quat'])
                )
                
                # Gripper Logic (0.5 threshold for boolean conversion if needed)
                grip_val = step['target_grip']
                if isinstance(grip_val, bool):
                    is_closed = grip_val
                else:
                    is_closed = grip_val > 0.5

                if is_closed:
                    gripper.grasp(0.1, 0.1) # Close
                else:
                    gripper.goto(255, 0.1, 0.1) # Open

                # 3. Record ACTUAL State (Physical Result)
                # This ensures the dataset contains valid physics states
                curr_pos_tensor, curr_quat_tensor = robot.get_ee_pose()
                
                # Check actual gripper state from Sim
                # If SIM.gripper_val is small (near 0), it's closed
                actual_grip_bool = SIM.gripper_val < 10.0

                # Schema must match the teleop episodes, which create_lmdb_*.py reads as:
                #   action  <- position / gripper           (what was commanded)
                #   proprio <- proc_pos / proc_gripper      (what the robot actually reached)
                # Recording only the achieved pose left proc_* missing, so process_episode
                # raised KeyError into its bare except and silently dropped every episode.
                # Per-step tilt of both adjacent blocks, sampled from the SAME rollout that
                # is being rendered. This is what a linear probe needs: re-simulating later
                # cannot recover it, because reset_simulation perturbs block pose with
                # np.random.uniform and saves neither the seed nor the initial qpos, so a
                # fresh run is a different trajectory than these frames.
                # Alignment is free: these ride on the waypoint's own timestamp, which is
                # the key create_lmdb_single30.py already matches frames against.
                with SIM.lock:
                    tilt_l = SIM.get_block_tilt("block_left")
                    tilt_r = SIM.get_block_tilt("block_right")
                    # target block: excluded from check_failure (it is meant to move), but
                    # recorded so an alarm driven by its motion can be distinguished from a
                    # genuine false positive
                    tilt_m = SIM.get_block_tilt("block_middle")
                    mx, my = SIM.get_block_xy("block_middle")

                actual_trajectory_data.append({
                    'timestamp': time.time(),
                    'position': np.asarray(step['target_pos']).tolist(),
                    'orientation': np.asarray(step['target_quat']).tolist(),
                    'gripper': bool(is_closed),
                    'proc_pos': curr_pos_tensor.numpy().tolist(),
                    'proc_quat': curr_quat_tensor.numpy().tolist(),
                    'proc_gripper': actual_grip_bool,
                    'tilt_left': round(float(tilt_l), 4),
                    'tilt_right': round(float(tilt_r), 4),
                    'tilt_middle': round(float(tilt_m), 4),
                    'mid_xy': [round(mx, 5), round(my, 5)],
                })

                # Ground truth for THIS rollout, captured as it happens.
                # Without this the outcome is lost and has to be reconstructed later by
                # re-simulating from a fresh random reset -- which produces a different
                # rollout than the frames recorded here, so the label would not describe
                # the trajectory the monitor is scored on.
                with SIM.lock:
                    failed, blk, tilt = SIM.check_failure()
                if tilt > peak_tilt:
                    peak_tilt, peak_block = float(tilt), blk or peak_block
                if failed and failure_step is None:
                    failure_step = len(actual_trajectory_data) - 1
                    failure_timestamp = actual_trajectory_data[-1]['timestamp']
                    failure_block = blk

                # (No preview: show_preview belongs to cam.py's real-camera Recorder,
                # not SimRecorder. Calling it here raised AttributeError, which the
                # outer try/except swallowed -- so every episode failed silently and
                # wrote zero frames.)

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
                    'noise_params': {'pos_std': NOISE_POS_STD, 'rot_std': NOISE_ROT_STD},
                    # Ground truth for exactly this rollout -- no re-simulation needed.
                    # failure_step indexes `waypoints`; failure_timestamp lets the LMDB
                    # builder re-align it after timestamp matching drops any waypoints.
                    'outcome': 'failure' if failure_step is not None else 'success',
                    'failure_step': failure_step,
                    'failure_timestamp': failure_timestamp,
                    'failure_block': failure_block,
                    'peak_tilt_deg': round(peak_tilt, 3),
                    'peak_block': peak_block,
                    'topple_threshold_deg': TOPPLE_THRESHOLD_DEG,
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