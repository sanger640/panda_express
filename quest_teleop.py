import asyncio
import websockets
import json
import torch
import numpy as np
from polymetis import RobotInterface
import ssl 

class QuestRobotTeleop:
    def __init__(self):
        print("Connecting to robot...")
        self.robot = RobotInterface(ip_address="localhost", port=50051)

        # Start impedance controller with soft compliance
        Kx = torch.Tensor([150., 150., 150., 10., 10., 10.])
        Kxd = torch.Tensor([10., 10., 10., 1., 1., 1.])
        self.robot.start_cartesian_impedance(Kx=Kx, Kxd=Kxd)

        print("Robot impedance controller started!")

        # Get initial robot pose
        self.initial_robot_pos, self.initial_robot_quat = self.robot.get_ee_pose()
        print(f"Initial robot EE position: {self.initial_robot_pos.numpy()}")

        # Calibration variables (set on first controller message)
        self.initial_controller_pos = None
        self.initial_controller_quat = None
        self.calibrated = False

        # Scaling factor (adjust to control sensitivity)
        self.position_scale = 1.0

        # Last gripper state
        self.last_gripper_closed = False

    def calibrate(self, controller_pos, controller_quat):
        """Calibrate coordinate systems on first message"""
        self.initial_controller_pos = controller_pos
        self.initial_controller_quat = controller_quat
        self.calibrated = True
        print(f"Calibrated! Controller initial position: {controller_pos}")
        print("Move your controller to control the robot.")

    async def handle_controller_data(self, websocket):
        """Process incoming controller data"""
        print("Quest controller connected! Waiting for calibration...")

        async for message in websocket:
            try:
                data = json.loads(message)

                # Extract controller data
                controller_pos = np.array(data['position'])
                controller_quat = np.array(data['orientation'])  # [x, y, z, w]
                trigger_value = data.get('trigger', 0.0)

                # Calibrate on first message
                if not self.calibrated:
                    self.calibrate(controller_pos, controller_quat)
                    continue

                # Calculate position delta from calibration point
                pos_delta = (controller_pos - self.initial_controller_pos) * self.position_scale

                # Apply delta to robot's initial position
                target_pos = self.initial_robot_pos.numpy() + pos_delta
                target_pos_tensor = torch.Tensor(target_pos)

                # Keep orientation fixed (can be enhanced later)
                target_quat_tensor = self.initial_robot_quat

                # Update robot pose
                self.robot.update_desired_ee_pose(
                    position=target_pos_tensor,
                    orientation=target_quat_tensor
                )

                # Handle gripper trigger
                gripper_closed = trigger_value > 0.5
                if gripper_closed != self.last_gripper_closed:
                    if gripper_closed:
                        print("Gripper CLOSED")
                    else:
                        print("Gripper OPEN")
                    self.last_gripper_closed = gripper_closed

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
            except Exception as e:
                print(f"Error processing controller data: {e}")

    async def start_server(self, host="0.0.0.0", port=8765):
        """Start WebSocket server"""
        print(f"Starting WebSocket server on {host}:{port}")
        async with websockets.serve(self.handle_controller_data, host, port):
            print("Server ready! Open the WebXR page on your Quest 3.")
            await asyncio.Future()  # Run forever

    def cleanup(self):
        """Cleanup robot connection"""
        print("\nStopping controller...")
        self.robot.terminate_current_policy()
        print("Done!")

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

async def main():
    teleop = QuestRobotTeleop()
    # Note the 'ssl=ssl_context' parameter
    async with websockets.serve(teleop.handle_controller_data, '0.0.0.0', 8765, ssl=ssl_context):
        print('WSS server running on port 8765')
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
