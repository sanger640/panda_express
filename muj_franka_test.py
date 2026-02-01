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

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)


        self.lock = threading.Lock()
        self.running = True
        
        # Launch viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Physics thread
        self.thread = threading.Thread(target=self._physics_loop, daemon=True)
        self.thread.start()

    def _physics_loop(self):
        last_render_time = 0
        target_fps = 30.0  # Cap the viewer at 30 FPS
        
        while self.running:
            start = time.time()
            with self.lock:
                mujoco.mj_step(self.model, self.data)
                
                # Only update viewer if enough time has passed (1/30 seconds)
                now = time.time()
                if now - last_render_time > (1.0 / target_fps):
                    if self.viewer.is_running():
                        self.viewer.sync()
                    last_render_time = now
            
            # Sync physics to real time
            elapsed = time.time() - start
            if elapsed < self.model.opt.timestep:
                time.sleep(self.model.opt.timestep - elapsed)
                
class SimRobotInterface:
    def __init__(self, ip_address=None):
        print(f"[SIM] Connected to MuJoCo Robot")
        with SIM.lock:
            # Snap mocap to current EE pos
            SIM.data.mocap_pos[0] = SIM.data.site_xpos[SIM.model.site('attachment_site').id]
            SIM.data.mocap_quat[0] = SIM.data.site_xmat[SIM.model.site('attachment_site').id]

    def get_ee_pose(self):
        with SIM.lock:
            pos = SIM.data.site_xpos[SIM.model.site('attachment_site').id].copy()
            mat = SIM.data.site_xmat[SIM.model.site('attachment_site').id].reshape(3, 3)
            quat = R.from_matrix(mat).as_quat() # x, y, z, w
            return torch.Tensor(pos), torch.Tensor(quat)

    def update_desired_ee_pose(self, position, orientation):
        with SIM.lock:
            if isinstance(position, torch.Tensor): position = position.numpy()
            if isinstance(orientation, torch.Tensor): orientation = orientation.numpy()
            
            SIM.data.mocap_pos[0] = position
            # MuJoCo uses scalar-first (w, x, y, z)
            wxyz = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
            SIM.data.mocap_quat[0] = wxyz

    def start_cartesian_impedance(self, Kx=None, Kxd=None):
        pass

    def terminate_current_policy(self):
        pass

# --- GRIPPER INTERFACE ---
class SimGripperInterface:
    def __init__(self, ip_address=None):
        pass
    
    def grasp(self, speed, force):
        with SIM.lock:
            SIM.data.ctrl[7] = 0.0 # Close
            SIM.data.ctrl[8] = 0.0

    def goto(self, width, speed, force):
        with SIM.lock:
            val = width / 2.0
            SIM.data.ctrl[7] = val
            SIM.data.ctrl[8] = val
            
    def stop(self):
        pass

# --- CAMERA & RECORDER ---

class SimDualCamera:
    """Used by client.py for inference (Low Res, RGB, Channel First)"""
    def __init__(self, cam1_serial=None, cam2_serial=None, H=240, W=320, hz=30):
        self.H = H
        self.W = W
        self.renderer = mujoco.Renderer(SIM.model, height=H, width=W)
    
    def get_frames(self):
        with SIM.lock:
            self.renderer.update_scene(SIM.data, camera="cam_wrist")
            img1 = self.renderer.render()
            self.renderer.update_scene(SIM.data, camera="cam_fixed")
            img2 = self.renderer.render()

        # Resize/Format for Policy: (C, H, W)
        img1 = cv2.resize(img1, (self.W, self.H)) # Should already be correct size from renderer
        img2 = cv2.resize(img2, (self.W, self.H))
        
        # Renderer returns RGB, Policy expects RGB channel first
        return np.transpose(img1, (2, 0, 1)), np.transpose(img2, (2, 0, 1))

class SimRecorder:
    """Used by teleop.py for recording data (High Res, BGR, Save to Disk)"""
    def __init__(self, i, cam1_serial, cam2_serial, ssd_loc):
        self.i = i
        # Mimic the folder structure in cam.py
        self.save_folder = os.path.join(ssd_loc, "episodes", str(self.i), "rgb_frames")
        os.makedirs(self.save_folder, exist_ok=True)
        
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2000)
        
        # High Res Renderer (640x480 matches real config)
        self.renderer = mujoco.Renderer(SIM.model, height=480, width=640)
        self.preview_frame = None

    def start(self):
        if not self.running:
            self.running = True
            
            # Start threads
            self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
            
            self.capture_thread.start()
            self.save_thread.start()
            print(f"[SIM] Recording started. Saving to: {self.save_folder}")

    def stop(self):
        self.running = False
        if hasattr(self, 'capture_thread'): self.capture_thread.join()
        if hasattr(self, 'save_thread'): self.save_thread.join()
        print("[SIM] Recording stopped.")

    def show_preview(self, is_recording=False):
        # Optional: Show a CV2 window of what is being recorded
        if self.preview_frame is not None:
            cv2.imshow("Sim Preview", self.preview_frame)
            cv2.waitKey(1)

    def _capture_worker(self):
        while self.running:
            start_time = time.time()
            
            with SIM.lock:
                self.renderer.update_scene(SIM.data, camera="cam_wrist")
                img1_rgb = self.renderer.render()
                self.renderer.update_scene(SIM.data, camera="cam_fixed")
                img2_rgb = self.renderer.render()

            # Convert RGB (MuJoCo) to BGR (OpenCV standard) for saving
            img1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2BGR)
            
            timestamp = time.time()
            self.preview_frame = np.hstack((img1, img2)) # For visualization

            try:
                self.frame_queue.put_nowait((timestamp, img1, img2))
            except queue.Full:
                pass
            
            # Maintain roughly 30Hz
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0/30.0) - elapsed)
            time.sleep(sleep_time)

    def _save_worker(self):
        while self.running or not self.frame_queue.empty():
            try:
                timestamp, img1, img2 = self.frame_queue.get(timeout=0.1)
                
                # Naming convention matches cam.py
                fname1 = os.path.join(self.save_folder, f"cam1_{int(timestamp * 1000)}.png")
                cv2.imwrite(fname1, img1)
                
                fname2 = os.path.join(self.save_folder, f"cam2_{int(timestamp * 1000)}.png")
                cv2.imwrite(fname2, img2)
                
                self.frame_queue.task_done()
            except queue.Empty:
                pass

def main():
    SIM = SimContext()

    while SIM.viewer.is_running():
            time.sleep(0.1)

if __name__ == "__main__":
    main()