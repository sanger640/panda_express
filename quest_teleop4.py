import asyncio
import websockets
import json
import torch
import numpy as np
from polymetis import RobotInterface, GripperInterface
from scipy.spatial.transform import Rotation as R
import ssl
import time
import grpc  # for RpcError checks
import pyrealsense2 as rs
import cv2
import os
import threading
import queue

MAX_POSITION_STEP = 0.02   # smaller step for safety
WORKSPACE_RADIUS = 0.25    # slightly smaller workspace
WORKSPACE_MIN_Z = 0.08
ROBOT_BASE = np.array([0.0, 0.0, 0.0])

class RealSenseRGBRecorder:
    def __init__(self, save_folder="rgb_frames"):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

        self.frame_queue = queue.Queue(maxsize=2000)  # buffer to store frames before saving
        self.running = False  # start stopped

    def start(self):
        if not self.running:
            self.running = True
            self.saving_thread = threading.Thread(target=self._saving_worker, daemon=True)
            self.saving_thread.start()
            self.rgb_thread = threading.Thread(target=self.capture_frame)
            self.rgb_thread.start()

    def capture_frame(self):
        while self.running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return
            img = np.asanyarray(color_frame.get_data())
            timestamp = time.time()
            self.latest_rgb_timestamp = timestamp
            try:
                self.frame_queue.put_nowait((timestamp, img))
            except queue.Full:
                print("[WARNING] Frame queue full, dropping frame")

    def _saving_worker(self):
        while self.running or not self.frame_queue.empty():
            try:
                timestamp, img = self.frame_queue.get(timeout=0.1)
                filename = os.path.join(self.save_folder, f"{int(timestamp * 1000)}.png")
                cv2.imwrite(filename, img)
                self.frame_queue.task_done()
            except queue.Empty:
                pass  # wait for frames

    def stop(self):
        if self.running:
            self.running = False
            self.saving_thread.join()
            self.rgb_thread.join()
            self.pipeline.stop()
        
class TeleopRecorder:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)
        self.rgb_running = False
        self.rgb_folder = "rgb_frames"
        os.makedirs(self.rgb_folder, exist_ok=True)
        self.latest_rgb_frame = None
        self.latest_rgb_timestamp = None

        # Start RGB capture thread
        self.rgb_thread = None
        # self.rgb_thread.start()
        # self.rgb_thread = None

        # For storing EE poses (timestamped)
        self.ee_poses = []

    def capture_rgb_frames(self):
        while self.rgb_running:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            img = np.asanyarray(color_frame.get_data())
            timestamp = time.time()
            filename = os.path.join(self.rgb_folder, f"{int(timestamp*1000)}.png")
            cv2.imwrite(filename, img)
            self.latest_rgb_frame = img
            self.latest_rgb_timestamp = timestamp
            time.sleep(1/30)  # enforce ~30Hz

    def record_ee_pose(self, pose):
        timestamp = time.time()
        self.ee_poses.append({'timestamp': timestamp, 'pose': pose})
        # Optionally associate nearest RGB frame timestamp here

    def stop(self):
        self.rgb_running = False
        self.rgb_thread.join()
        self.pipeline.stop()
        # Save pose data to file here if needed

class AdvancedQuestRobotTeleop:
    """Advanced teleoperation with position AND orientation control"""

    def __init__(self, enable_orientation=False):
        print("Connecting to robot...")
        self.robot = RobotInterface(ip_address="129.97.71.27")
        self.gripper = GripperInterface(ip_address="129.97.71.27")
        # Control flags
        self.enable_orientation = enable_orientation

        # Start impedance controller (softer gains)
        # self.Kx = torch.Tensor([160, 160, 160, 15, 15, 15])
        # self.Kxd = torch.Tensor([15, 15, 15, 2, 2, 2])
        self.Kx = torch.Tensor([750, 750, 750, 15, 15, 15])
        self.Kxd = torch.Tensor([37, 37, 37, 2, 2, 2])
        self.robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
        print("Robot impedance controller started!")

        # Get initial robot pose
        self.initial_robot_pos, self.initial_robot_quat = self.robot.get_ee_pose()
        print(f"Initial robot EE position: {self.initial_robot_pos.numpy()}")
        print(f"Orientation control: {'ENABLED' if enable_orientation else 'DISABLED'}")

        # Calibration variables
        self.initial_controller_pos = None
        self.initial_controller_rot = None
        self.initial_robot_rot = None
        self.calibrated = False
        self.prev_target_pos = None
        self.prev_target_quat = None

        # Control parameters (slow & gentle)
        self.position_scale = 2.0
        self.rotation_scale = 0.3  # kept for future use if you scale rotation

        # For control-rate throttling (~10 Hz)
        self.last_command_time = 0.0
        self.command_period = 0.1

        # For velocity limiting / dt
        self.prev_time = time.time()
        self.max_velocity = 0.1  # 5 cm/s EE velocity

        # For input change detection
        self.prev_controller_pos = None
        self.prev_controller_quat = None

        # Deadband parameters
        self.DEADBAND_POS = 0.01  # 1 cm deadband in robot-frame controller pos
        self.DEADBAND_EE = 0.005  # 5 mm deadband on EE target position
        self.DEADBAND_ORI_RAD = np.deg2rad(3)  # 3 degrees deadband on EE orientation

        # Recording
        self.recording = False
        self.trajectory_data = []
        self.record_start_time = None
        self._grip_pressed  = False
        # Gripper state
        self.last_gripper_closed = False

        # Initialize recorder but do not start RGB thread yet
        self.recorder = RealSenseRGBRecorder()

    def transform_controller_to_robot(self, pos):
        # Map: Robot X = -Controller Z, Robot Y = -Controller X, Robot Z = Controller Y
        return np.array([-pos[0], pos[2], pos[1]])

    def transform_controller_quat(self, q):
        # Input: [x, y, z, w] (controller)
        # Output: [z, x, y, w] (robot) with axis swap and sign flips
        return np.array([-q[0], q[2], q[1], q[3]])

    def calibrate(self, controller_pos, controller_quat):
        """Calibrate coordinate systems"""
        self.initial_controller_pos = self.transform_controller_to_robot(controller_pos)
        self.initial_controller_rot = R.from_quat(self.transform_controller_quat(controller_quat))
        self.initial_robot_rot = R.from_quat(self.initial_robot_quat.numpy())
        self.prev_controller_pos = controller_pos.copy()
        self.prev_controller_quat = controller_quat.copy()
        self.calibrated = True
        self.prev_target_pos = self.initial_robot_pos.numpy()
        self.prev_target_quat = self.initial_robot_quat.numpy()

        print(f"\n{'='*50}")
        print("CALIBRATED!")
        print(f"Controller position: {controller_pos}")
        print(f"Robot EE position: {self.initial_robot_pos.numpy()}")
        print(f"{'='*50}\n")
        print("Ready to teleoperate! Move your controller.")

    async def handle_controller_data(self, websocket):
        """Process incoming controller data"""
        print("\nQuest controller connected!")
        print("Waiting for first controller input to calibrate...")

        async for message in websocket:
            try:
                # --- Throttle control rate to ~10 Hz ---
                now = time.time()
                if now - self.last_command_time < self.command_period:
                    continue
                self.last_command_time = now

                data = json.loads(message)

                # Extract controller data
                controller_pos = np.array(data['position'])
                controller_quat = np.array(data['orientation'])
                trigger_value = data.get('trigger', 0.0)
                grip_button = data.get('grip', 0.0)
                # print("controller pos")
                # print(controller_pos)
                # --- Input change detection (position) ---
                # if self.prev_controller_pos is not None:
                #     pos_change = np.linalg.norm(controller_pos - self.prev_controller_pos)
                #     if pos_change > 0.3:  # 10 cm sudden jump
                #         print(f"[WARNING] Large controller position jump: {pos_change:.3f} m. Skipping this frame.")
                #         self.prev_controller_pos = controller_pos.copy()
                #         self.prev_controller_quat = controller_quat.copy()
                #         continue
                # self.prev_controller_pos = controller_pos.copy()
                # self.prev_controller_quat = controller_quat.copy()

                # Calibrate on first message
                if not self.calibrated:
                    self.calibrate(controller_pos, controller_quat)
                    continue

                # === POSITION DEADBAND ON CONTROLLER INPUT ===
                controller_pos_robot = self.transform_controller_to_robot(controller_pos)
                delta_ctrl = np.linalg.norm(controller_pos_robot - self.prev_controller_pos) if self.prev_controller_pos is not None else np.inf
                if delta_ctrl < self.DEADBAND_POS:
                    # Use last controller pos in robot frame
                    controller_pos_robot = self.prev_controller_pos.copy()

                # Calculate position delta and EE target position
                pos_delta = (controller_pos_robot - self.initial_controller_pos) * self.position_scale
                target_pos = self.initial_robot_pos.numpy() + pos_delta

                # # === LOW PASS FILTER ON POSITION ===
                # alpha = 0.1
                # if self.prev_target_pos is not None:
                #     target_pos = alpha * target_pos + (1.0 - alpha) * self.prev_target_pos

                # # Clamp workspace boundaries
                # vec_from_base = target_pos - ROBOT_BASE
                # norm_dist = np.linalg.norm(vec_from_base)
                # if norm_dist > WORKSPACE_RADIUS:
                #     target_pos = ROBOT_BASE + (vec_from_base / norm_dist) * WORKSPACE_RADIUS
                #     print("[SAFEGUARD] Target outside workspace sphere, clamped.")
                # if target_pos[2] < WORKSPACE_MIN_Z:
                #     target_pos[2] = WORKSPACE_MIN_Z
                #     print("[SAFEGUARD] Target z below table, clamped.")

                # === DEAD BAND ON EE TARGET POSITION ===
                if self.prev_target_pos is not None:
                    delta_ee = np.linalg.norm(target_pos - self.prev_target_pos)
                    if delta_ee < self.DEADBAND_EE:
                        target_pos = self.prev_target_pos.copy()



                # === VELOCITY LIMITING ===
                dt = now - self.prev_time
                if dt > 0 and self.prev_target_pos is not None:
                    dist = np.linalg.norm(target_pos - self.prev_target_pos)
                    vel = dist / dt
                    if vel > self.max_velocity and dist > 1e-6:
                        direction = (target_pos - self.prev_target_pos) / dist
                        target_pos = self.prev_target_pos + direction * self.max_velocity * dt
                        print(f"[VELOCITY LIMIT] Clamped EE speed to {self.max_velocity:.3f} m/s")
                self.prev_time = now

                # Clamp position step
                # step = np.linalg.norm(target_pos - self.prev_target_pos)
                # if step > MAX_POSITION_STEP:
                #     # direction = (target_pos - self.prev_target_pos) / step
                #     # target_pos = self.prev_target_pos + direction * MAX_POSITION_STEP
                #     target_pos = self.prev_target_pos

                #     print(f"[SAFEGUARD] Position step too large ({step:.3f} m), clamped.")



                self.prev_target_pos = target_pos.copy()
                target_pos_tensor = torch.Tensor(target_pos)

                # === ORIENTATION CONTROL WITH DEADBAND ===
                if self.enable_orientation:
                    current_controller_rot = R.from_quat(self.transform_controller_quat(controller_quat))
                    delta_rot = current_controller_rot * self.initial_controller_rot.inv()
                    # target_rot = delta_rot * self.initial_robot_rot
                    # target_quat = target_rot.as_quat()
                    
                    # # Smooth orientation lightly
                    # if self.prev_target_quat is not None:
                    #     beta = 0.1
                    #     blended = beta * target_quat + (1.0 - beta) * self.prev_target_quat
                    #     blended = blended / np.linalg.norm(blended)
                    #     target_quat = blended

                    angle = delta_rot.magnitude()  # rotation angle in radians
                    if angle < self.DEADBAND_ORI_RAD and self.prev_target_quat is not None:
                        target_quat = self.prev_target_quat.copy()
                    else:
                        target_rot = delta_rot * self.initial_robot_rot
                        target_quat = target_rot.as_quat()

                    
                    self.prev_target_quat = target_quat.copy()

                    target_quat_tensor = torch.Tensor(target_quat)
                else:
                    target_quat_tensor = self.initial_robot_quat

                self.prev_controller_pos = controller_pos.copy()
                self.prev_controller_quat = controller_quat.copy()
                # print("target pos")
                # print(target_pos)
                # Update robot pose with error recovery
                try:
                    self.robot.update_desired_ee_pose(
                        position=target_pos_tensor,
                        orientation=target_quat_tensor
                    )
                except grpc.RpcError as e:
                    msg = str(e)
                    if (
                        "no controller running" in msg
                        or "power_limit_violation" in msg
                        or "Safety limits exceeded" in msg
                    ):
                        print(f"[ERROR] {msg}. Restarting Cartesian impedance controller...")
                        self.robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
                        self.robot.update_desired_ee_pose(
                            position=target_pos_tensor,
                            orientation=target_quat_tensor
                        )
                    else:
                        raise e

                # self.recorder.capture_frame()
                # Gripper control
                gripper_closed = trigger_value > 0.5
                if gripper_closed != self.last_gripper_closed:
                    if gripper_closed:
                        print("Gripper Closed")
                        self.gripper.grasp(speed=0.05, force=0.1)
                    else:
                        print("Gripper Open")
                        self.gripper.stop()
                        self.gripper.goto(width=0.25, speed=0.05, force=0.1)
                    self.last_gripper_closed = gripper_closed
                # Recording toggle
                # if grip_button > 0.5:
                #     self._grip_pressed = not self._grip_pressed
                #     print("grip button test")
                #     self.toggle_recording()
                # Grip button logic to start/stop recording
                if grip_button > 0.5 and not self._grip_pressed:
                    self._grip_pressed = True
                    self.recording = True
                    self.trajectory_data = []
                    self.record_start_time = time.time()
                    self.recorder.start()
                    print("Grip pressed. Starting recording.")
                elif grip_button <= 0.5 and self._grip_pressed:
                    self._grip_pressed = False
                    self.recording = False
                    self.save_trajectory()
                    self.recorder.stop()
                    print("Grip released. Stopping recording.")
                    

                if self.recording:
                    if self.enable_orientation:
                        self.record_waypoint(target_pos_tensor, target_quat_tensor, gripper_closed)
                    else:
                        self.record_waypoint(target_pos_tensor, self.initial_robot_quat, gripper_closed)
                # Record pose at 10Hz as before
                # pose_dict = {
                #     'position': target_pos,
                #     'orientation': target_quat_tensor.numpy() if self.enable_orientation else self.initial_robot_quat.numpy()
                # }
                # self.recorder.record_ee_pose(pose_dict)

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except Exception as e:
                print(f"Error processing data: {e}")
                import traceback
                traceback.print_exc()

    def toggle_recording(self):
        if not self.recording:
            self.recording = True
            self.trajectory_data = []
            self.record_start_time = asyncio.get_event_loop().time()
            print("\n🔴 RECORDING STARTED")
        else:
            self.recording = False
            print(f"\n⏹ RECORDING STOPPED ({len(self.trajectory_data)} waypoints)")
            self.save_trajectory()
    
    def start_recording(self):
        self.trajectory_data = []
        self.record_start_time = time.time()
        # Start RealSense RGB capture thread
        if not self.recorder.rgb_running:
            self.recorder.rgb_running = True
            self.recorder.rgb_thread = threading.Thread(target=self.recorder.capture_rgb_frames)
            self.recorder.rgb_thread.start()
        self.recording = True
        print("\n🔴 RECORDING STARTED")
    def stop_recording(self):
        # Stop RealSense RGB capture thread
        if self.recorder.rgb_running:
            self.recorder.rgb_running = False
            self.recorder.rgb_thread.join()
            self.recorder.pipeline.stop()
        self.recording = False
        self.save_trajectory()
        print(f"\n⏹ RECORDING STOPPED ({len(self.trajectory_data)} waypoints)")
        
        

    def record_waypoint(self, position, orientation, gripper):
        timestamp = time.time()
        self.trajectory_data.append({
            'timestamp': timestamp,
            'position': position.numpy().tolist(),
            'orientation': orientation.numpy().tolist(),
            'gripper': gripper
        })

    def save_trajectory(self):
        if not self.trajectory_data:
            print("No data to save!")
            return

        import time as _time
        filename = f"trajectory_{int(_time.time())}.json"

        trajectory = {
            'metadata': {
                'num_waypoints': len(self.trajectory_data),
                'duration': self.trajectory_data[-1]['timestamp'],
                'position_scale': self.position_scale,
                'orientation_enabled': self.enable_orientation
            },
            'waypoints': self.trajectory_data
        }

        with open(filename, 'w') as f:
            json.dump(trajectory, f, indent=2)

        print(f"✓ Saved trajectory to: {filename}")

    async def start_server(self, host="0.0.0.0", port=8765):
        print(f"\nStarting WebSocket server on {host}:{port}")
        async with websockets.serve(self.handle_controller_data, host, port):
            print("Server ready!")
            print("\nInstructions:")
            print("1. Open WebXR page on Quest 3")
            print("2. Enter VR mode")
            print("3. Move controller to calibrate")
            print("4. Press GRIP button to start/stop recording")
            print("\nWaiting for Quest connection...")
            await asyncio.Future()

    def cleanup(self):
        print("\nCleaning up...")
        if self.recording:
            self.save_trajectory()
        self.robot.terminate_current_policy()
        print("Done!")

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

async def main():
    teleop = AdvancedQuestRobotTeleop(enable_orientation=False)
    async with websockets.serve(teleop.handle_controller_data, '0.0.0.0', 8765, ssl=ssl_context):
        print('WSS server running on port 8765')
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
