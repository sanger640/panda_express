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
        
        # Launch viewer (non-blocking)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.thread.start()

    def reset_simulation(self):
        """Resets MuJoCo data and applies 5% random noise to blocks."""
        with self.lock:
            # 1. Standard Reset to Keyframe
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

            # 2. Define the blocks we want to randomize
            block_names = ["block_middle", "block_left", "block_right"]
            
            # Noise parameters
            # pos_noise_range = 0.005
            # rot_noise_range = 8.0

            pos_noise_range = 0.005
            rot_noise_range = 3.0 

            for name in block_names:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id == -1: continue
                
                joint_id = self.model.body_jntadr[body_id]
                qpos_adr = self.model.jnt_qposadr[joint_id]
                
                if qpos_adr != -1:
                    # Apply X, Y variation
                    self.data.qpos[qpos_adr] += np.random.uniform(-pos_noise_range, pos_noise_range)
                    self.data.qpos[qpos_adr + 1] += np.random.uniform(-pos_noise_range, pos_noise_range)

                    # Apply Z-axis rotation variation
                    orig_quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
                    z_angle = np.radians(np.random.uniform(-rot_noise_range, rot_noise_range))
                    noise_rot = R.from_euler('z', z_angle)
                    
                    # MuJoCo is WXYZ, Scipy is XYZW
                    curr_rot = R.from_quat([orig_quat[1], orig_quat[2], orig_quat[3], orig_quat[0]])
                    new_rot = noise_rot * curr_rot
                    new_quat = new_rot.as_quat()
                    
                    # Back to WXYZ
                    self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

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
                # 1. RUN CONTROLLER
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
        
        self.kp_pos = 500.0   
        # self.kp_pos = 150
        self.kd_pos = 30.0
        # self.kd_pos = 14   
        self.kp_rot = 20.0
        # self.kp_rot = 10.0
        self.kd_rot = 0.5     
        self.knull  = 0.1     

        with SIM.lock:
            self.site_id = SIM.model.site('attachment_site').id
            SIM.gripper_val = GP
            
            # Set initial target to current pose
            self.target_pos = SIM.data.site_xpos[self.site_id].copy()
            self.target_quat = np.zeros(4)
            mujoco.mju_mat2Quat(self.target_quat, SIM.data.site_xmat[self.site_id])
            print("hey there")
            # REGISTER CONTROLLER
            SIM.control_callback = self._internal_control_loop

    def _internal_control_loop(self):
        """Runs inside SimContext lock every physics step."""
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
        """Thread-safe update of the target."""
        if isinstance(position, torch.Tensor): position = position.numpy()
        if isinstance(orientation, torch.Tensor): orientation = orientation.numpy()
        
        with SIM.lock:
            self.target_pos = position
            # Convert scalar-last (scipy) to scalar-first (mujoco)
            self.target_quat = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])

    def get_ee_pose(self):
        with SIM.lock:
            pos = SIM.data.site_xpos[self.site_id].copy()
            mat = SIM.data.site_xmat[self.site_id].reshape(3, 3)
            quat = R.from_matrix(mat).as_quat() 
            return torch.Tensor(pos), torch.Tensor(quat)

    # --- ADDED METHODS FOR COMPATIBILITY ---
    def get_state(self):
        """Returns [x, y, z, grip_state] where grip_state is 1.0 (closed) or -1.0 (open)."""
        pos, _ = self.get_ee_pose()
        pos = pos.numpy()
        
        # Map Sim gripper (0=Closed, 110=Open) to Policy (1=Closed, -1=Open)
        is_closed = (SIM.gripper_val < 50.0) 
        grip_state = [1.0] if is_closed else [-1.0]
        
        return np.concatenate([pos, grip_state]).astype(np.float32)

    def execute(self, action):
        """Executes a 4D action [x, y, z, grip_cmd]."""
        # 1. Parse Action
        target_pos = torch.from_numpy(action[:3]).float()
        grip_cmd = action[3]
        
        # 2. Keep current orientation (convert internal WXYZ back to XYZW for update method)
        curr_quat_wxyz = self.target_quat
        target_quat = torch.tensor([curr_quat_wxyz[1], curr_quat_wxyz[2], curr_quat_wxyz[3], curr_quat_wxyz[0]])
        
        # 3. Update Robot Target
        self.update_desired_ee_pose(target_pos, target_quat)
        
        # 4. Handle Gripper
        if grip_cmd > 0.9:
            with SIM.lock: SIM.gripper_val = 0.0 # Close
        elif grip_cmd < -0.9:
            with SIM.lock: SIM.gripper_val = GP # Open

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