import numpy as np
from polymetis import RobotInterface, GripperInterface
import torch

class Franka:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        print(f"Connecting to Polymetis at {self.robot_ip}...")
        self.robot = RobotInterface(ip_address=self.robot_ip)
        self.gripper = GripperInterface(ip_address=self.robot_ip)
        self.prev_target_pos = None
        self.initial_robot_rot = None
        self.gripper_is_closed = False
        self.Kx = torch.Tensor([750, 750, 750, 15, 15, 15])
        self.Kxd = torch.Tensor([37, 37, 37, 2, 2, 2])
        self.robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
        init_pos, _ = self.robot.get_ee_pose()
        self.last_cmd_pos = (init_pos.numpy())
        print("Robot Connected & In Impedance Mode.")

    def get_state(self):
        pos, quat = self.robot.get_ee_pose()
        pos = pos.numpy()
        quat = quat.numpy() # [x, y, z, w]

        if self.initial_robot_rot is None:
            self.initial_robot_rot = quat
        # 4D State: [x, y, z, grip_cmd]
        grip_state = [1.0] if self.gripper_is_closed else [-1.0] # 1=Closed, -1=Open

        return np.concatenate([self.last_cmd_pos, grip_state]).astype(np.float32)
        
    def execute(self, action):
        # Action: [x, y, z, grip]
        target_pos = torch.Tensor(action[:3])
        # Fixed Down Orientation
        # target_quat = torch.Tensor([1.0, 0.0, 0.0, 0.0]) 
        grip_cmd = action[3]
        self.last_cmd_pos = action[:3]
        self.robot.update_desired_ee_pose(position=target_pos, orientation=torch.Tensor(self.initial_robot_rot))
        
        if grip_cmd > 0.9:
            if not self.gripper_is_closed:
                self.gripper.grasp(speed=0.05, force=0.1)
                self.gripper_is_closed = True
        elif grip_cmd < -0.9:
            if self.gripper_is_closed:
                self.gripper.stop()
                self.gripper.goto(width=0.25, speed=0.05, force=0.1)
                self.gripper_is_closed = False
        else:
            pass