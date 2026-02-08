import mujoco
import mujoco.viewer
import numpy as np
import threading
import queue
import time
import cv2
import os
from scipy.spatial.transform import Rotation as R
import torch

class SimContext:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("franka_emika_panda/panda_jenga_setup.xml")
        self.data = mujoco.MjData(self.model)

        # Shared state
        self.gripper_val = 0.0
        
        # Callback for the robot controller (The "Brain")
        self.control_callback = None

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        self.lock = threading.Lock()
        self.running = True
        
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.thread.start()

    # def reset_simulation(self):
    #     """Resets the MuJoCo data to the initial keyframe safely"""
    #     with self.lock:
    #         # Reset to keyframe 0 (defined in your XML)
    #         mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
            
    #         # Ensure the gripper state in the simulation matches the 'Open' constant
    #         self.gripper_val = GP
            
    #         # Re-run forward kinematics to update site positions immediately
    #         mujoco.mj_forward(self.model, self.data)
    #         print("[SIM] Scene reset to keyframe 0")

    def reset_simulation(self):
        """Resets MuJoCo data and applies 5% random noise to blocks."""
        with self.lock:
            # 1. Standard Reset to Keyframe
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            # 2. Define the blocks we want to randomize
            block_names = ["block_middle", "block_left", "block_right"]
            
            # Workspace scale for '5%' (based on table/block layout ~0.4m scale)
            # 5% of 0.4m is approx 0.02m (2cm)
            pos_noise_range = 0.005
            # 5% of 180 degrees is approx 9 degrees
            rot_noise_range = 8.0 

            for name in block_names:
                # print(name)
                # Find the qpos address for the body (free joint)
                # qpos for free joint is 7 elements: [x, y, z, qw, qx, qy, qz]
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                # print("bod id")
                # print(body_id)
                joint_id = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[joint_id]
                # print("qpos_addy")
                # print(qpos_adr)
                if qpos_adr != -1:
                    # Apply X, Y variation (indices 0 and 1)
                    # print("0")
                    # print(self.data.qpos[qpos_adr])
                    self.data.qpos[qpos_adr] += np.random.uniform(-pos_noise_range, pos_noise_range)
                    # print("1")
                    # print(self.data.qpos[qpos_adr+1])
                    self.data.qpos[qpos_adr + 1] += np.random.uniform(-pos_noise_range, pos_noise_range)

                    # Apply XY (Z-axis) rotation variation
                    # Get current orientation
                    orig_quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
                    
                    # Generate random rotation around Z axis
                    z_angle = np.radians(np.random.uniform(-rot_noise_range, rot_noise_range))
                    noise_rot = R.from_euler('z', z_angle)
                    
                    # Combine orientations (MuJoCo is WXYZ, Scipy is XYZW)
                    curr_rot = R.from_quat([orig_quat[1], orig_quat[2], orig_quat[3], orig_quat[0]])
                    new_rot = noise_rot * curr_rot
                    new_quat = new_rot.as_quat()
                    
                    # Back to MuJoCo WXYZ
                    self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]
                    # Verification Print
                    pos = self.data.qpos[qpos_adr : qpos_adr + 3]
                    print(f"[{name}] BodyID: {body_id} | QPos Adr: {qpos_adr} | New Pos: {pos.round(4)}")

            # 3. Finalize
            self.gripper_val = GP
            mujoco.mj_forward(self.model, self.data)
            print("[SIM] Scene reset with random variation applied.")

    def _physics_loop(self):
        last_render_time = 0
        target_fps = 30.0
        
        while self.running:
            start = time.time()
            with self.lock:
                # 1. RUN CONTROLLER (If registered)
                # This ensures torques are updated every single physics step!
                if self.control_callback is not None:
                    self.control_callback()
                
                # 2. STEP PHYSICS
                mujoco.mj_step(self.model, self.data)
                
                # 3. RENDER
                now = time.time()
                if now - last_render_time > (1.0 / target_fps):
                    if self.viewer.is_running():
                        self.viewer.sync()
                    last_render_time = now
            
            elapsed = time.time() - start
            if elapsed < self.model.opt.timestep:
                time.sleep(self.model.opt.timestep - elapsed)

SIM = SimContext()
GP = 110
class SimRobotInterface:
    def __init__(self, ip_address=None):
        print(f"[SIM] Connected to MuJoCo Robot (Background OSC Mode)")
        
        self.kp_pos = 150.0   
        self.kd_pos = 14.0    
        self.kp_rot = 10.0    
        self.kd_rot = 0.5     
        self.knull  = 0.1     

        with SIM.lock:
            self.site_id = SIM.model.site('attachment_site').id
            SIM.gripper_val = GP
            
            # Set initial target to current pose to prevent jumping
            self.target_pos = SIM.data.site_xpos[self.site_id].copy()
            self.target_quat = np.zeros(4)
            mujoco.mju_mat2Quat(self.target_quat, SIM.data.site_xmat[self.site_id])
            
            # REGISTER CONTROLLER to the background thread
            SIM.control_callback = self._internal_control_loop

    def _internal_control_loop(self):
        """
        This runs inside the SimContext lock, every physics step.
        It keeps the robot floating at self.target_pos.
        """
        # A. Current State
        curr_pos = SIM.data.site_xpos[self.site_id]
        curr_mat = SIM.data.site_xmat[self.site_id].reshape(3, 3)
        
        # B. Position Error
        dx = self.target_pos - curr_pos
        
        # C. Orientation Error
        target_mat = np.zeros(9)
        mujoco.mju_quat2Mat(target_mat, self.target_quat)
        target_mat = target_mat.reshape(3, 3)
        
        rot_err_mat = target_mat @ curr_mat.T
        dr = np.array([
            rot_err_mat[2, 1] - rot_err_mat[1, 2],
            rot_err_mat[0, 2] - rot_err_mat[2, 0],
            rot_err_mat[1, 0] - rot_err_mat[0, 1]
        ]) * 0.5

        # D. Velocity & Force
        J = np.zeros((6, SIM.model.nv))
        mujoco.mj_jacSite(SIM.model, SIM.data, J[:3], J[3:], self.site_id)
        vel_cartesian = J @ SIM.data.qvel
        
        F_pos = self.kp_pos * dx - self.kd_pos * vel_cartesian[:3]
        F_rot = self.kp_rot * dr - self.kd_rot * vel_cartesian[3:]
        F_wrench = np.hstack([F_pos, F_rot])
        
        # E. Torques
        tau_task = J.T @ F_wrench
        tau_null = -self.knull * SIM.data.qvel
        tau_total = tau_task + tau_null
        
        # F. Apply
        SIM.data.ctrl[:7] = tau_total[:7]
        SIM.data.ctrl[7] = SIM.gripper_val

    def update_desired_ee_pose(self, position, orientation):
        """
        Now simply updates the target. The background thread handles the physics.
        """
        if isinstance(position, torch.Tensor): position = position.numpy()
        if isinstance(orientation, torch.Tensor): orientation = orientation.numpy()
        # position = np.array([position[1], -position[0], position[2]])
        # Thread-safe update of the target
        with SIM.lock:
            self.target_pos = position
            # Convert scalar-last (scipy) to scalar-first (mujoco)
            self.target_quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])

    def get_ee_pose(self):
        with SIM.lock:
            pos = SIM.data.site_xpos[self.site_id].copy()
            # pos = np.array([pos[1], -pos[0], pos[2]])
            mat = SIM.data.site_xmat[self.site_id].reshape(3, 3)
            quat = R.from_matrix(mat).as_quat() 
            return torch.Tensor(pos), torch.Tensor(quat)

    def reset(self):
        """Interface method called by Teleop"""
        # 1. Trigger the physics reset
        SIM.reset_simulation()
        
        # 2. Crucial: Update the internal target_pos to the new reset position
        # This prevents the robot from 'flying' back to where it was before the reset
        with SIM.lock:
            self.target_pos = SIM.data.site_xpos[self.site_id].copy()
            
            # Ensure self.target_quat is float64 and writeable
            if self.target_quat.dtype != np.float64:
                self.target_quat = self.target_quat.astype(np.float64)
                
            # Call with explicit float64 views
            mujoco.mju_mat2Quat(self.target_quat, SIM.data.site_xmat[self.site_id].astype(np.float64))

    def start_cartesian_impedance(self, Kx=None, Kxd=None):
        pass

    def terminate_current_policy(self):
        pass

# --- GRIPPER & CAMERA CLASSES REMAIN UNCHANGED ---
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
        self.renderer = mujoco.Renderer(SIM.model, height=480, width=640)
        self.preview_frame = None

    def show_preview(self, is_recording=False):
        if self.preview_frame is not None:
            # Draw a red circle if recording
            if is_recording:
                cv2.circle(self.preview_frame, (30, 30), 10, (0, 0, 255), -1)
            
            cv2.imshow("Sim Preview", self.preview_frame)
            cv2.waitKey(1)

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
        if hasattr(self, 'save_thread'): self.save_thread.join()

    # def _capture_worker(self):
    #     while self.running:
    #         start = time.time()
    #         with SIM.lock:
    #             self.renderer.update_scene(SIM.data, camera="cam_wrist")
    #             img1 = self.renderer.render()
    #             self.renderer.update_scene(SIM.data, camera="cam_fixed")
    #             img2 = self.renderer.render()
    #         img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    #         img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    #         try: self.frame_queue.put_nowait((time.time(), img1, img2))
    #         except queue.Full: pass
    #         time.sleep(max(0, (1.0/30.0) - (time.time()-start)))

    def _capture_worker(self):
        while self.running:
            start_time = time.time()
            
            # --- CAMERA 1 ---
            # 1. Lock only to SNAP the data (Fast)
            with SIM.lock:
                self.renderer.update_scene(SIM.data, camera="cam_wrist")
            
            # 2. Render the pixels (Slow) - DO THIS UNLOCKED
            # This allows physics to keep running while we draw!
            img1_rgb = self.renderer.render()

            # --- CAMERA 2 ---
            # 3. Lock only to SNAP the data (Fast)
            with SIM.lock:
                self.renderer.update_scene(SIM.data, camera="cam_fixed")
            
            # 4. Render the pixels (Slow) - DO THIS UNLOCKED
            img2_rgb = self.renderer.render()

            # ... The rest of the processing remains the same ...
            img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
            
            self.preview_frame = np.hstack((img1, img2))

            try:
                self.frame_queue.put_nowait((time.time(), img1, img2))
            except queue.Full:
                pass
            
            # Maintain roughly 30Hz
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0/30.0) - elapsed)
            time.sleep(sleep_time)

    def _save_worker(self):
        while self.running or not self.frame_queue.empty():
            try:
                ts, i1, i2 = self.frame_queue.get(timeout=0.1)
                cv2.imwrite(os.path.join(self.save_folder, f"cam1_{int(ts*1000)}.png"), i1)
                cv2.imwrite(os.path.join(self.save_folder, f"cam2_{int(ts*1000)}.png"), i2)
            except queue.Empty: pass

def main():
    while SIM.viewer.is_running(): time.sleep(0.1)

if __name__ == "__main__":
    main()