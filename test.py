import numpy as np
from polymetis import RobotInterface

def main():
    # use ethernet port of the host
    robot = RobotInterface(ip_address="129.97.71.27")

    target_position = np.array([0.4, 0.0, 0.4])
    target_orientation = np.array([0, 0, 0, 1])  # Quaternion (x,y,z,w)

    # Set target EE pose
    robot.set_ee_pose(target_position, target_orientation)

    # Run control loop until target reached or timeout
    for _ in range(1000):
        robot.step()  # send control commands
    robot.disconnect()

if __name__ == "__main__":
    main()
