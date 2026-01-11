import asyncio
import websockets
import json
import torch
import numpy as np
from polymetis import RobotInterface, GripperInterface
from scipy.spatial.transform import Rotation as R
import ssl
import time
import grpc
import os
import traceback

from cam import DualRealsenseRecorder
# import shutil

# --- config ---
MAX_POSITION_STEP = 0.02   # smaller step for safety
WORKSPACE_RADIUS = 0.25    # slightly smaller workspace
WORKSPACE_MIN_Z = 0.08
ROBOT_BASE = np.array([0.0, 0.0, 0.0])

CAMERA_1_SERIAL = "215222078938" 
CAMERA_2_SERIAL = "819612070440"
ROBOT_IP = "129.97.71.27"
SSD_LOC="/mnt/diffusion_policy/tasks/pick_place/dual_wrist/"
# ---------------------

class Teleop:
    """Advanced teleoperation with position AND orientation control"""

    def __init__(self, enable_orientation=False):
        self.robot = RobotInterface(ip_address=ROBOT_IP)
        self.gripper = GripperInterface(ip_address=ROBOT_IP)
        self.enable_orientation = enable_orientation

        # impd contr
        self.Kx = torch.Tensor([750, 750, 750, 15, 15, 15])
        self.Kxd = torch.Tensor([37, 37, 37, 2, 2, 2])
        self.robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
        print("Controller started")

        # Get init robot pose
        self.initial_robot_pos, self.initial_robot_quat = self.robot.get_ee_pose()
        print(f"Initial robot EE position: {self.initial_robot_pos.numpy()}")
        print(f"Orientation control: {'ENABLED' if self.enable_orientation else 'DISABLED'}")

        # Calibration variables
        self.initial_controller_pos = None
        self.initial_controller_rot = None
        self.initial_robot_rot = None
        self.prev_target_pos = None
        self.prev_target_quat = None
        self.calibrated = False
        self.prev_controller_pos = None
        self.prev_controller_quat = None


        # controller to robot scaling
        self.position_scale = np.array([1.2, 1.2, 2.2])
        self.rotation_scale = 0.3  

        # limit controller output (from 60hz) to 10Hz
        self.last_command_time = 0.0
        self.command_period = 0.1

        # set max vel of robot movement
        self.prev_time = time.time()
        self.max_velocity = 0.2  # 5 cm/s EE velocity

        # deadband params to filter out tiny controller movement
        self.DEADBAND_POS = 0.01  
        self.DEADBAND_EE = 0.005  
        self.DEADBAND_ORI_RAD = np.deg2rad(3) 

        # initially not recording
        self.recording = False
        self.trajectory_data = []
        self.record_start_time = None
        self._grip_pressed  = False

        # gripper state (initially open)
        self.last_gripper_closed = False

        # episode number to record from (and init recorder)
        self.i = 1
        self.recorder = DualRealsenseRecorder(self.i, CAM1_SERIAL, CAM2_SERIAL, SSD_LOC)

        # dont track controller at start
        self.track = False
        self.pos_delta = 0

    # transform controller values into robot frame
    def transform_controller_to_robot(self, pos):
        # in : [x, y z] (controller frame)
        # out : [-x, z, y] (robot frame)
        return np.array([-pos[0], pos[2], pos[1]])

    def transform_controller_quat(self, q):
        # in: [x, y, z, w] (controller frame)
        # out: [-x, z, y, w] (robot frame) 
        return np.array([-q[0], q[2], q[1], q[3]])

    def calibrate(self, controller_pos, controller_quat):
        """Calibrate coordinate systems"""
        # get contr val in robot frame
        self.initial_controller_pos = self.transform_controller_to_robot(controller_pos)
        self.initial_controller_rot = R.from_quat(self.transform_controller_quat(controller_quat))
        # get init robot rot mat
        self.initial_robot_rot = R.from_quat(self.initial_robot_quat.numpy())
        # set prev cont pose/rot same as init val (timestep 0)
        self.prev_controller_pos = controller_pos.copy()
        self.prev_controller_quat = controller_quat.copy()
        # set target pos as intial robot pos (timestep 0)
        self.prev_target_pos = self.initial_robot_pos.numpy()
        self.prev_target_quat = self.initial_robot_quat.numpy()
        self.calibrated = True
        
        print("CALIBRATED!")
        print(f"Controller position: {controller_pos}")
        print(f"Robot EE position: {self.initial_robot_pos.numpy()}")
        print("Ready to teleoperate! Move your controller.")
    
    def tracking(self, a_butt, b_butt, controller_pos):
        # if a button pressed then rob starts tracking the contr, if b pressed stop tracking
        if a_butt > 0.5 and self.track == False:
            self.track = True
            controller_pos_robot = self.transform_controller_to_robot(controller_pos)
            # to reinitialize the init_control_pos so pos_delta remains same as the last tracked timestep
            self.initial_controller_pos = controller_pos_robot - (self.pos_delta/self.position_scale)
        if b_butt > 0.5 and self.track == True:
            self.track = False

    def deadband(self, controller_pos, controller_quat):
        # deadband filter on contr input (need prev and curr control val)
        controller_pos_robot = self.transform_controller_to_robot(controller_pos)
        delta_ctrl = np.linalg.norm(controller_pos_robot - self.prev_controller_pos) if self.prev_controller_pos is not None else np.inf
        if delta_ctrl < self.DEADBAND_POS:
            # use prev control pose val (basically don't move)
            controller_pos_robot = self.prev_controller_pos.copy()

        # calc ee target pos to reach by contr (need init timestep 0 cont pos)
        self.pos_delta = (controller_pos_robot - self.initial_controller_pos) * self.position_scale
        target_pos = self.initial_robot_pos.numpy() + self.pos_delta

        # deadband on orientation (not tested)
        if self.enable_orientation:
            current_controller_rot = R.from_quat(self.transform_controller_quat(controller_quat))
            delta_rot = current_controller_rot * R.from_quat(self.transform_controller_quat(self.prev_controller_quat)).inv()
            
            angle = delta_rot.magnitude()  # rotation angle in radians
            if angle < self.DEADBAND_ORI_RAD and self.prev_target_quat is not None:
                target_quat = self.prev_target_quat.copy()
            else:
                target_rot = delta_rot * self.initial_robot_rot
                target_quat = target_rot.as_quat()
            
            self.prev_target_quat = target_quat.copy()
            # target_quat_tensor = torch.Tensor(target_quat)
        else:
            target_quat = self.initial_robot_quat

        return target_pos, target_quat

    def clamp_vel(self, now, target_pos):
        # check ee vel is safe, need prev targ pos and prev time
        dt = now - self.prev_time
        if dt > 0 and self.prev_target_pos is not None:
            dist = np.linalg.norm(target_pos - self.prev_target_pos)
            vel = dist / dt
            if vel > self.max_velocity and dist > 1e-6:
                # clamp vel
                direction = (target_pos - self.prev_target_pos) / dist
                target_pos = self.prev_target_pos + direction * self.max_velocity * dt
        self.prev_time = now
        self.prev_target_pos = target_pos.copy()
        return target_pos

    def execute(self, target_pos_tensor, target_quat_tensor):
        # error recov (bad need to improve, remove for now or comment it out, need to calibrate upon recovery)
        try:
            self.robot.update_desired_ee_pose(
                position=target_pos_tensor,
                orientation=target_quat_tensor
            )
        except grpc.RpcError as e:
            msg = str(e)
            # implement better logic for safety limits exceeded
            if "no controller running" in msg or "power_limit_violation" in msg or "Safety limits exceeded" in msg:
                print(f"[ERROR] {msg}. Restarting Cartesian impedance controller...")
                self.robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
                self.robot.update_desired_ee_pose(
                    position=target_pos_tensor,
                    orientation=target_quat_tensor
                )
            else:
                raise e

    def gripper_action(self, trigger_value):
        gripper_closed = trigger_value > 0.5
        # see if grip val diff then last timestep
        if gripper_closed != self.last_gripper_closed:
            if gripper_closed:
                print("Gripper Closed")
                self.gripper.grasp(speed=0.05, force=0.1)
            else:
                print("Gripper Open")
                self.gripper.stop() # if was closing, stop it and open
                self.gripper.goto(width=0.25, speed=0.05, force=0.1)
            self.last_gripper_closed = gripper_closed

    def recording(self):
        pass

    
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
            print("No data")
            return

        filename = f"{SSD_LOC}episodes/{self.i}/trajectory_{int(_time.time())}.json"

        trajectory = {
            'metadata': {
                'num_waypoints': len(self.trajectory_data),
                'duration': self.trajectory_data[-1]['timestamp'],
                'orientation_enabled': self.enable_orientation
            },
            'waypoints': self.trajectory_data
        }

        with open(filename, 'w') as f:
            json.dump(trajectory, f, indent=2)

        print(f"Saved trajectory to: {filename}")
        self.i +=1

    async def handle_controller_data(self, websocket):
        """Process incoming controller data"""
        print("Quest controller connected!")
        print("Press 'A' to start moving")
        print("Press 'B' to stop moving")

        async for message in websocket:
            try:
                # throttle to 10hz
                now = time.time()
                if now - self.last_command_time < self.command_period:
                    continue
                self.last_command_time = now

                data = json.loads(message)

                # get data from controller
                controller_pos = np.array(data['position'])
                controller_quat = np.array(data['orientation'])
                trigger_value = data.get('trigger', 0.0)
                grip_button = data.get('grip', 0.0)
                a_butt = data.get('button_a', 0.0)
                b_butt = data.get('button_b', 0.0)

                self.tracking(a_butt, b_butt, controller_pos)

                if self.track == True: 

                    # calibrate on first message
                    if not self.calibrated:
                        self.calibrate(controller_pos, controller_quat)
                        continue

                    target_pos, target_quat = self.deadband(controller_pos)
                    
                    # deadband on ee target pos (not needed i think); need prev target pos
                    # if self.prev_target_pos is not None:
                    #     delta_ee = np.linalg.norm(target_pos - self.prev_target_pos)
                    #     if delta_ee < self.DEADBAND_EE:
                    #         target_pos = self.prev_target_pos.copy()

                    target_pos = self.clamp_vel(now, target_pos)
                    # conv to tensors
                    target_pos_tensor = torch.Tensor(target_pos)
                    target_quat_tensor=torch.Tensor(target_quat)
                    self.prev_controller_pos = controller_pos.copy()
                    self.prev_controller_quat = controller_quat.copy()

                    self.execute(target_pos_tensor, target_quat_tensor)
                    
                    # Gripper control
                    self.gripper_action(trigger_value)
                    self.recording(grip_button)

                    # Recording control (use diff button name, create a func)
                    # check whether to start or stop rec
                    if grip_button > 0.5 and not self._grip_pressed:
                        self._grip_pressed = True
                        self.recording = True
                        self.trajectory_data = []
                        self.record_start_time = time.time()
                        self.recorder.start()
                        print(f"Grip pressed. Starting recording. Episode {self.i}/{self.recorder.i}")
                    elif grip_button <= 0.5 and self._grip_pressed:
                        self._grip_pressed = False
                        self.recording = False
                        self.save_trajectory()
                        self.recorder.stop()
                        self.recorder.i += 1
                        self.recorder.save_folder = SSD_LOC+"episodes/" + str(self.i) + "/rgb_frames/"
                        os.makedirs(self.recorder.save_folder, exist_ok=True)
                        print(f"RECORDING STOPPED ({len(self.trajectory_data)} waypoints)")

                    if self.recording:
                        # record waypoints
                        if self.enable_orientation:
                            self.record_waypoint(target_pos_tensor, target_quat_tensor, gripper_closed)
                        else:
                            self.record_waypoint(target_pos_tensor, self.initial_robot_quat, gripper_closed)

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
            except Exception as e:
                print(f"Error processing data: {e}")
                traceback.print_exc()



ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(certfile='cert.pem', keyfile='key.pem')

async def main():
    teleop = Teleop(enable_orientation=False)
    async with websockets.serve(teleop.handle_controller_data, '0.0.0.0', 8765, ssl=ssl_context):
        print('WSS server running on port 8765')
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())