import os

# 1. Force the Hardware Backend BEFORE MuJoCo loads
os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import mujoco.viewer
import numpy as np
import threading
import queue
import time
import cv2
from scipy.spatial.transform import Rotation as R
import torch
import mujoco.gl_context

# A block is "toppled" only once it is lying flat, not merely leaning. Measured peak
# tilt across episodes is strongly bimodal: blocks either stay under ~16 deg (standing,
# possibly nudged) or go to ~95 deg (flat), with nothing in between. Any threshold in
# 20-75 deg yields the same verdict; 45 sits in the middle of that dead zone.
# See survey_tilts.py and RESUME.md 5b.
TOPPLE_THRESHOLD_DEG = 45.0


class SimContext:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("franka_emika_panda/panda_jenga_setup.xml")
        self.data = mujoco.MjData(self.model)

        # Shared state
        self.gripper_val = 0.0
        self.control_callback = None

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.lock = threading.Lock()
        self.running = True

        self._ref_quats = {}
        self._ref_axis_tilt = {}
        self._record_ref_quats()  # store upright reference before any noise is applied

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.thread.start()

    def reset_simulation(self):
        with self.lock:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            block_names = ["block_middle", "block_left", "block_right"]
            pos_noise_range = 0.002
            rot_noise_range = 3.0 

            for name in block_names:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id == -1: continue
                
                joint_id = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[joint_id]
                
                if qpos_adr != -1:
                    self.data.qpos[qpos_adr] += np.random.uniform(-pos_noise_range, pos_noise_range)
                    self.data.qpos[qpos_adr + 1] += np.random.uniform(-pos_noise_range, pos_noise_range)

                    orig_quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
                    z_angle = np.radians(np.random.uniform(-rot_noise_range, rot_noise_range))
                    noise_rot = R.from_euler('z', z_angle)
                    
                    curr_rot = R.from_quat([orig_quat[1], orig_quat[2], orig_quat[3], orig_quat[0]])
                    new_rot = noise_rot * curr_rot
                    new_quat = new_rot.as_quat()
                    
                    self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

            self.gripper_val = GP
            mujoco.mj_forward(self.model, self.data)
            self._record_ref_quats()  # update reference after each reset (new random pose)
            print("[SIM] Scene reset with random variation applied.")

    def _record_ref_quats(self):
        """Store post-reset reference state for the two adjacent blocks (left and right).

        Called after every reset so tilts are measured relative to the actual starting pose,
        not the XML default. Records both the reference quaternion (kept for reference) and
        the reference tilt-from-vertical, which is what get_block_tilt actually uses."""
        for name in ["block_left", "block_right"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id == -1:
                continue
            joint_id = self.model.body_jntadr[body_id]
            qpos_adr = self.model.jnt_qposadr[joint_id]
            if qpos_adr != -1:
                self._ref_quats[name] = self.data.qpos[qpos_adr + 3:qpos_adr + 7].copy()
            self._ref_axis_tilt[name] = self._axis_tilt(name)

    def _axis_tilt(self, block_name):
        """Absolute angle between the block's own z-axis and world z, in degrees.

        This is the physically meaningful notion of "tipping": it ignores rotation about the
        vertical (yaw), so a block that slides or spins while staying upright reads ~0. The
        reset noise is pure yaw (see reset_simulation), so this measure is unaffected by it."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, block_name)
        if body_id == -1:
            return 0.0
        rot = self.data.xmat[body_id].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(rot[2, 2], -1.0, 1.0))))

    def get_block_tilt(self, block_name):
        """Returns how far the block has tipped from vertical since the last reset, in degrees.

        Measured as tilt-from-vertical rather than full quaternion distance, so pure yaw
        (a block sliding or spinning while still standing) does not count as tipping. The
        previous quaternion-geodesic version registered ~10 deg of spurious tilt on blocks
        that never left vertical, which mattered when the threshold was 15 deg."""
        if block_name not in self._ref_axis_tilt:
            return 0.0
        return abs(self._axis_tilt(block_name) - self._ref_axis_tilt[block_name])

    def check_failure(self, threshold_deg=TOPPLE_THRESHOLD_DEG):
        """Check whether any adjacent block has toppled beyond threshold_deg.

        Returns:
            failed (bool): True if any block has toppled.
            block_name (str|None): Name of the toppled block, or None.
            tilt_deg (float): Worst tilt angle observed across both blocks.
        """
        worst_tilt, worst_block = 0.0, None
        for name in ["block_left", "block_right"]:
            tilt = self.get_block_tilt(name)
            if tilt > worst_tilt:
                worst_tilt, worst_block = tilt, name
        if worst_tilt > threshold_deg:
            return True, worst_block, worst_tilt
        return False, None, worst_tilt

    def _physics_loop(self):
        last_render_time = time.time()
        target_fps = 30.0

        # Reset internal simulation time for synchronization
        with self.lock:
            self.data.time = 0.0
        time_zero = time.time()
        last_sim_time = 0.0  # Add this tracker
        
        while self.running:
            with self.lock:
                if self.control_callback is not None:
                    self.control_callback()
                
                mujoco.mj_step(self.model, self.data)
                
                now = time.time()
                if now - last_render_time > (1.0 / target_fps):
                    if self.viewer.is_running():
                        self.viewer.sync()
                    last_render_time = now
            
            # --- THE FIX: CATCH-UP REAL-TIME LOOP ---
            # This calculates exactly where the simulation *should* be in real-world time.
            # If we are ahead, it calls time.sleep() which yields the GIL so the camera thread can run.
            # If OS timers oversleep, it just skips the sleep and blasts through physics steps to catch up.
            expected_real_time = time_zero + self.data.time
            current_time = time.time()
            delay = expected_real_time - current_time
            if delay > 0:
                time.sleep(delay)
    def _physics_loop(self):
        last_render_time = time.time()
        # Viewer sync rate (lowering this to 15.0 or 20.0 can give the camera thread more GPU airtime if needed)
        target_fps = 30.0 
        
        # Reset internal simulation time for synchronization
        with self.lock:
            self.data.time = 0.0
            
        time_zero = time.time()
        last_sim_time = 0.0  # You added this...
        
        while self.running:
            with self.lock:
                if self.control_callback is not None:
                    self.control_callback()
                
                mujoco.mj_step(self.model, self.data)
                
                # --- THE TIME TRAVEL FIX (Added this block) ---
                # If time went backward, a reset happened. Re-anchor to current real-world time!
                if self.data.time < last_sim_time:
                    time_zero = time.time() - self.data.time
                last_sim_time = self.data.time
                # ----------------------------------------------
                
                now = time.time()
                if now - last_render_time > (1.0 / target_fps):
                    if hasattr(self, 'viewer') and self.viewer.is_running():
                        self.viewer.sync()
                    last_render_time = now
            
            # --- CATCH-UP REAL-TIME LOOP ---
            expected_real_time = time_zero + self.data.time
            current_time = time.time()
            delay = expected_real_time - current_time
            if delay > 0:
                time.sleep(delay)
SIM = SimContext()
GP = 110

class SimRobotInterface:
    def __init__(self, ip_address=None):
        print(f"[SIM] Connected to MuJoCo Robot")
        
        self.kp_pos = 500.0   
        self.kd_pos = 30.0
        self.kp_rot = 20.0
        self.kd_rot = 0.5     
        self.knull  = 0.1     

        with SIM.lock:
            self.site_id = SIM.model.site('attachment_site').id
            SIM.gripper_val = GP
            
            self.target_pos = SIM.data.site_xpos[self.site_id].copy()
            self.target_quat = np.zeros(4)
            mujoco.mju_mat2Quat(self.target_quat, SIM.data.site_xmat[self.site_id])
            SIM.control_callback = self._internal_control_loop

    def _internal_control_loop(self):
        curr_pos = SIM.data.site_xpos[self.site_id]
        curr_mat = SIM.data.site_xmat[self.site_id].reshape(3, 3)
        
        dx = self.target_pos - curr_pos
        
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, self.target_quat)
        target_mat = target_mat.reshape(3, 3)
        
        rot_err_mat = target_mat @ curr_mat.T
        dr = np.array([
            rot_err_mat[2, 1] - rot_err_mat[1, 2],
            rot_err_mat[0, 2] - rot_err_mat[2, 0],
            rot_err_mat[1, 0] - rot_err_mat[0, 1]
        ]) * 0.5

        J = np.zeros((6, SIM.model.nv))
        mujoco.mj_jacSite(SIM.model, SIM.data, J[:3], J[3:], self.site_id)
        vel_cartesian = J @ SIM.data.qvel
        
        F_pos = self.kp_pos * dx - self.kd_pos * vel_cartesian[:3]
        F_rot = self.kp_rot * dr - self.kd_rot * vel_cartesian[3:]
        F_wrench = np.hstack([F_pos, F_rot])
        
        tau_task = J.T @ F_wrench
        tau_null = -self.knull * SIM.data.qvel
        tau_total = tau_task + tau_null
        
        SIM.data.ctrl[:7] = tau_total[:7]
        SIM.data.ctrl[7] = SIM.gripper_val

    def update_desired_ee_pose(self, position, orientation):
        if isinstance(position, torch.Tensor): position = position.numpy()
        if isinstance(orientation, torch.Tensor): orientation = orientation.numpy()
        
        with SIM.lock:
            self.target_pos = position
            self.target_quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])

    def get_ee_pose(self):
        with SIM.lock:
            pos = SIM.data.site_xpos[self.site_id].copy()
            mat = SIM.data.site_xmat[self.site_id].reshape(3, 3)
            quat = R.from_matrix(mat).as_quat() 
            return torch.Tensor(pos), torch.Tensor(quat)

    def get_state(self):
        pos, _ = self.get_ee_pose()
        pos = pos.numpy()
        is_closed = (SIM.gripper_val < 50.0) 
        grip_state = [1.0] if is_closed else [-1.0]
        return np.concatenate([pos, grip_state]).astype(np.float32)

    def execute(self, action):
        target_pos = torch.from_numpy(action[:3]).float()
        grip_cmd = action[3]
        
        curr_quat_wxyz = self.target_quat
        target_quat = torch.tensor([curr_quat_wxyz[1], curr_quat_wxyz[2], curr_quat_wxyz[3], curr_quat_wxyz[0]])
        
        self.update_desired_ee_pose(target_pos, target_quat)
        
        if grip_cmd > 0.9:
            with SIM.lock: SIM.gripper_val = 0.0 
        elif grip_cmd < -0.9:
            with SIM.lock: SIM.gripper_val = GP 

    def reset(self):
        SIM.reset_simulation()
        with SIM.lock:
            self.target_pos = SIM.data.site_xpos[self.site_id].copy()
            if self.target_quat.dtype != np.float64:
                self.target_quat = self.target_quat.astype(np.float64)
            mujoco.mju_mat2Quat(self.target_quat, SIM.data.site_xmat[self.site_id])

    def start_cartesian_impedance(self, Kx=None, Kxd=None): pass
    def terminate_current_policy(self): pass

class SimGripperInterface:
    def __init__(self, ip_address=None): pass
    def grasp(self, speed, force):
        with SIM.lock: SIM.gripper_val = 0.0
    def goto(self, width, speed, force):
        with SIM.lock: SIM.gripper_val = GP
    def stop(self): pass

class SimDualCamera:
    def __init__(self, cam1_serial=None, cam2_serial=None, H=240, W=320, hz=30):
        self.H, self.W = H, W
        self.renderer = mujoco.Renderer(SIM.model, height=H, width=W)
        
    def get_frames(self):
        with SIM.lock:
            self.renderer.update_scene(SIM.data, camera="cam_wrist")
            img1 = self.renderer.render()
            self.renderer.update_scene(SIM.data, camera="cam_fixed")
            img2 = self.renderer.render()
        img1 = cv2.resize(img1, (self.W, self.H))
        img2 = cv2.resize(img2, (self.W, self.H))
        return np.transpose(img1, (2, 0, 1)), np.transpose(img2, (2, 0, 1))

class SimRecorder:
    def __init__(self, i, cam1_serial, cam2_serial, ssd_loc):
        self.i = i
        self.save_folder = os.path.join(ssd_loc, "episodes", str(self.i), "rgb_frames")
        os.makedirs(self.save_folder, exist_ok=True)
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2000)

    def start(self):
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
            self.capture_thread.start()
            self.save_thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'capture_thread'): self.capture_thread.join()
        # while(not self.frame_queue.empty()):
        #     pass
        if hasattr(self, 'save_thread'): self.save_thread.join()

    def _capture_worker(self):
        renderer_wrist = mujoco.Renderer(SIM.model, height=480, width=640)
        renderer_fixed = mujoco.Renderer(SIM.model, height=480, width=640)
        
        target_fps = 30.0
        period = 1.0 / target_fps
        
        # Initialize OUTSIDE the loop
        next_time = time.time() 

        while self.running:
            # Rigidly step forward exactly 33.3ms
            next_time += period 
            
            with SIM.lock:
                renderer_wrist.update_scene(SIM.data, camera="cam_wrist")
                renderer_fixed.update_scene(SIM.data, camera="cam_fixed")

            img1_rgb = renderer_wrist.render()
            img2_rgb = renderer_fixed.render()

            try:
                self.frame_queue.put_nowait((time.time(), img1_rgb, img2_rgb))
            except queue.Full:
                pass
            
            # Sleep logic remains the same
            now = time.time()
            sleep_time = next_time - now
            
            if sleep_time > 0.005: 
                time.sleep(sleep_time - 0.005)
            
            while time.time() < next_time:
                pass

    def _save_worker(self):
        fast_png = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        
        while self.running or not self.frame_queue.empty():
            try:
                ts, i1_rgb, i2_rgb = self.frame_queue.get(timeout=0.1)

                i1 = cv2.cvtColor(i1_rgb, cv2.COLOR_RGB2BGR)
                i2 = cv2.cvtColor(i2_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.save_folder, f"cam1_{int(ts*1000)}.png"), i1, fast_png)
                cv2.imwrite(os.path.join(self.save_folder, f"cam2_{int(ts*1000)}.png"), i2, fast_png)
            except queue.Empty: pass

def main():
    while SIM.viewer.is_running(): time.sleep(0.1)

if __name__ == "__main__":
    main()