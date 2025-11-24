import asyncio
import websockets
import json
import torch
import numpy as np
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation as R
import ssl 

MAX_POSITION_STEP = 0.05  # meters per update (tune this for your control frequency)
WORKSPACE_RADIUS = 0.3   # max(m), adjust to your robot
WORKSPACE_MIN_Z = 0.08   # min z for tabletop
ROBOT_BASE = np.array([0.0, 0.0, 0.0])  # or your base frame

class AdvancedQuestRobotTeleop:
    """Advanced teleoperation with position AND orientation control"""

    def __init__(self, enable_orientation=False):
        print("Connecting to robot...")
        self.robot = RobotInterface(ip_address="129.97.71.27")

        # Control flags
        self.enable_orientation = enable_orientation

        # Start impedance controller
        Kx = torch.Tensor([750, 750, 750, 15, 15, 15])
        Kxd = torch.Tensor([37, 37, 37, 2, 2, 2])
        self.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)

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

        # Control parameters
        self.position_scale = 0.005
        self.rotation_scale = 0.005

        # Recording
        self.recording = False
        self.trajectory_data = []
        self.record_start_time = None

        # Gripper state
        self.last_gripper_closed = False

        # Store controller gains for restarts
        self.Kx = Kx
        self.Kxd = Kxd

    def transform_controller_to_robot(self, pos):
        # Map: Robot X = -Controller Z, Robot Y = -Controller X, Robot Z = Controller Y
        return np.array([-pos[2], -pos[0], pos[1]])
    
    def transform_controller_quat(self, q):
        # Input: [x, y, z, w] (controller)
        # Output: [z, x, y, w] (robot) with axis swap and sign flips
        return np.array([-q[2], -q[0], q[1], q[3]])

    def calibrate(self, controller_pos, controller_quat):
        """Calibrate coordinate systems"""
        self.initial_controller_pos = self.transform_controller_to_robot(controller_pos)
        self.initial_controller_rot = R.from_quat(self.transform_controller_quat(controller_quat))
        self.initial_robot_rot = R.from_quat(self.initial_robot_quat.numpy())
        self.calibrated = True
        self.prev_target_pos = self.initial_robot_pos.numpy()

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
                data = json.loads(message)

                # Extract controller data
                controller_pos = np.array(data['position'])
                controller_quat = np.array(data['orientation'])
                print(controller_quat)
                trigger_value = data.get('trigger', 0.0)
                grip_button = data.get('grip', 0.0)

                # Calibrate on first message
                if not self.calibrated:
                    self.calibrate(controller_pos, controller_quat)
                    continue

                # === POSITION CONTROL ===
                controller_pos_robot = self.transform_controller_to_robot(controller_pos)
                pos_delta = (controller_pos_robot - self.initial_controller_pos) * self.position_scale
                target_pos = self.initial_robot_pos.numpy() + pos_delta

                # Apply low-pass filter for smoothing
                alpha = 0.1
                if self.prev_target_pos is not None:
                    target_pos = alpha * target_pos + (1 - alpha) * self.prev_target_pos

                # Safeguard: Clamp too large position step
                step = np.linalg.norm(target_pos - self.prev_target_pos)
                if step > MAX_POSITION_STEP:
                    direction = (target_pos - self.prev_target_pos) / step
                    target_pos = self.prev_target_pos + direction * MAX_POSITION_STEP
                    print(f"[SAFEGUARD] Position step too large ({step:.3f} m), clamped.")

                # Safeguard: Clamp workspace boundaries
                vec_from_base = target_pos - ROBOT_BASE
                norm_dist = np.linalg.norm(vec_from_base)
                if norm_dist > WORKSPACE_RADIUS:
                    target_pos = ROBOT_BASE + (vec_from_base / norm_dist) * WORKSPACE_RADIUS
                    print("[SAFEGUARD] Target outside workspace sphere, clamped.")
                if target_pos[2] < WORKSPACE_MIN_Z:
                    target_pos[2] = WORKSPACE_MIN_Z
                    print("[SAFEGUARD] Target z below table, clamped.")

                self.prev_target_pos = target_pos.copy()
                target_pos_tensor = torch.Tensor(target_pos)

                # === ORIENTATION CONTROL ===
                if self.enable_orientation:
                    current_controller_rot = R.from_quat(self.transform_controller_quat(controller_quat))
                    delta_rot = current_controller_rot * self.initial_controller_rot.inv()
                    target_rot = delta_rot * self.initial_robot_rot
                    target_quat = target_rot.as_quat()
                    target_quat_tensor = torch.Tensor(target_quat)
                else:
                    target_quat_tensor = self.initial_robot_quat

                # === Update robot pose - try with controller restart if fails ===
                try:
                    self.robot.update_desired_ee_pose(
                        position=target_pos_tensor,
                        orientation=target_quat_tensor
                    )
                except Exception as e:
                    # Check for controller not running error and restart
                    if "no controller running" in str(e):
                        print("Controller error detected, restarting Cartesian impedance controller.")
                        self.robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
                        self.robot.update_desired_ee_pose(
                            position=target_pos_tensor,
                            orientation=target_quat_tensor
                        )
                    else:
                        raise e

                # === GRIPPER CONTROL ===
                gripper_closed = trigger_value > 0.5
                if gripper_closed != self.last_gripper_closed:
                    print(f"Gripper: {'CLOSED' if gripper_closed else 'OPEN'}")
                    self.last_gripper_closed = gripper_closed

                # === RECORDING ===
                if grip_button > 0.5 and not hasattr(self, '_grip_pressed'):
                    self._grip_pressed = True
                    self.toggle_recording()
                elif grip_button <= 0.5:
                    self._grip_pressed = False

                if self.recording:
                    self.record_waypoint(target_pos_tensor, target_quat_tensor, gripper_closed)

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except Exception as e:
                print(f"Error processing data: {e}")
                import traceback
                traceback.print_exc()

    def toggle_recording(self):
        """Toggle trajectory recording"""
        if not self.recording:
            self.recording = True
            self.trajectory_data = []
            self.record_start_time = asyncio.get_event_loop().time()
            print("\n🔴 RECORDING STARTED")
        else:
            self.recording = False
            print(f"\n⏹ RECORDING STOPPED ({len(self.trajectory_data)} waypoints)")
            self.save_trajectory()

    def record_waypoint(self, position, orientation, gripper):
        """Record a single waypoint"""
        timestamp = asyncio.get_event_loop().time() - self.record_start_time
        self.trajectory_data.append({
            'timestamp': timestamp,
            'position': position.numpy().tolist(),
            'orientation': orientation.numpy().tolist(),
            'gripper': gripper
        })

    def save_trajectory(self):
        """Save recorded trajectory to file"""
        if not self.trajectory_data:
            print("No data to save!")
            return

        import time
        filename = f"trajectory_{int(time.time())}.json"

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
        """Start WebSocket server"""
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
        """Cleanup"""
        print("\nCleaning up...")
        if self.recording:
            self.save_trajectory()
        self.robot.terminate_current_policy()
        print("Done!")

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

async def main():
    # Set enable_orientation=True to control both position and orientation
    # Set enable_orientation=False for position-only control
    teleop = AdvancedQuestRobotTeleop(enable_orientation=False)

    async with websockets.serve(teleop.handle_controller_data, '0.0.0.0', 8765, ssl=ssl_context):
        print('WSS server running on port 8765')
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
