
## Introduction
This repository implements the following functionalities:

- VR (Meta Quest 3 Pro) teleoperation for Franka Panda robot manipulator.
- Recording demonstrations (end effector pose, gripper values, and camera streams).
- Robot client to execute actions given by the diffusion policy on a remote server.

There is a Dockerfile provided (along with docker compose) to setup the enviornment to run the scripts.
This repository utilizes [Polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis) send realtime cartesian commands to the Franka Robot.

<div align="center">
  <img src="assets/trials.gif" width="48%" />
  <img src="assets/trials_move_cup.gif" width="48%" />
  <p><i>Diffusion Policy in action</i></p>
</div>

## Hardware Setup
- Robot: Frank Panda Emika Robot
    - IP: "129.97.71.19"
    - Change to your Robot IP in franka_hardware.yaml in Polymetis repo  
- Gripper: Standard Panda Gripper
    - IP: "129.97.71.19"
    - Change to your Gripper IP in franka_hand.yaml in Polymetis repo
- Dual Camera setup
    - 1 External Camera: D435 Realsense
    - 1 Wrist Camera: D435 Realsense

<div align="center">
  <img src="https://github.com/sanger640/panda_express/blob/master/assets/diff_policy_setup.jpg?raw=true" width="400">
  <p><i>Franka Panda Robot Setup</i></p>
</div>

## Episode Format
- The epsiodes are recorded in the following format.
- All the epsiodes are stored within `episodes/` directory.
    - Within this dir, each episode is stored within it's respective number
    - Within each episode, there is a `start_timestamp.json` file and a `rgb_frames` folder.
    - `start_timestamp.json` file contains the waypoints of the EE pose in the world frame along with gripper values and their corresponding timestamps.
        - This is recorded at 10 Hz.
    - `rgb_frames` folder contains rgb frames of the cameras recorded at 30 hz at 640x480 resolution. Each frame is labelled by their timestamp, and in the case of the dual setup they will have a prefix of `cam_1` or `cam_2`. 
- The episodes folder is converted into zarr format so that it can be loaded and trained via diffusion policy.


## Instructions

### Starting the VR Teleop
- Mount SSD to directly record demonstrations to it (check section for it).

- Build and run the docker container
```bash
DOCKER_BUILDKIT=0 docker compose build
DOCKER_BUILDKIT=0 docker compose up
```
- Open 4 Termainals and activate the conda env in all four terminals:
```bash
conda init
conda activate polymetis-local
```

- Terminal 1 : This connects to the robot control manager.
```bash
pkill -9 run_server # kill prev hanging process
launch_robot.py robot_client=franka_hardware ip=129.97.71.27
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.readonly=true ip=129.97.71.27 # run in read only mode, cannot execute control cmds.
```
- Terminal 2  : This connects to the gripper.
```bash
launch_gripper.py ip=129.97.71.27
```

- Terminal 3 :  Host the `quest_controller.html` server connects with the Meta Quest 3 Pro.
```bash
python host_server.py
```

- Terminal 4 : Run the teleop script which listens to the Meta Controller commands (controller pose + button commands) and outputs robot cartesian pose commands, gripper commands (open/close) and whether to start/end recording of a given demonstration (episode). 
```bash
python home.py # to reset the robot to a determined home pose
python quest_teleop4.py
```

- Connect to the hosted server on the Meta Quest Pro Browser and enter VR mode.
```bash
https://129.97.71.84:8443/quest_controller.html
```

<div align="center">
  <img src="assets/teleop_demo.gif" width="400">
  <p><i>Teleop Demonstration</i></p>
</div>

### Recording Demonstrations (only need right hand controller)

- After starting the VR Teleop, follow these steps to record demonstrations for policy training:

1. Put your right hand in the "home position" (shown in image below)
2. Press 'A' on the controller to start moving the robot arm (Press 'B' to stop the robot following the controller)
3. Press and hold the trigger button near the bottom of the controller to start recording the episode
4. Move the robot near the object
5. Press and hold the trigger button at the top to close the gripper to grasp the object
6. Depending on the task, pick and place the object to the desired location
7. After completion of task, move robot back to home position
8. Release the lower trigger button to finish recording the episode
9. Press 'B' to stop the robot from following the controller.
10. Reset the enviornment and start from 1. Do this until enough episodes are recorded.

**Tip:** If you have an undesirable episode, do not stop the program to delete it, just note down the episode number and continue. After recording all the epsiode, delete the undesirable episode directory.
**Tip:** Try to record diverse episodes, with diverse inital enviornment and object configs with different lightings, and multiple ways to carry out the task, including recovery action in the case of grasp failure.

### Executing policy actions

After training a diffusion policy based on the collected demonstrations, we deploy the policy using a client-server paradigm. We use the `robohub` server cluster to host the trained model for inference, and the Franka robot computer act as the client. The client queries the server for robot action, and it sends robot observations (camera images + robot state) to the server in that query. The robot server recieves the robot observations for the observation time horizon and outputs (and sends back) robot actions for the specified robot action time horizon. The client recieves the robot action and executes it, and this loop repeats. This feedback loop runs at 10hz.


Run the model_server (on robohub machine)
```bash
python diff_server.py
```

Run the robot client (on robot machine):
```bash
python diff_dual_client.py
```

Make sure you change the IP (in `diff_dual_client.py`) if you host the model server on your own machine.

### Mount SSD Volume
- First repair it (just in case)
```bash
sudo ntfsfix /dev/sda1
```
- Then mount on `/mnt/`
```bash
sudo mount /dev/sdb1 /mnt/ssd_data
ls /mnt/ssd_data
```
- Then attach to docker compose (already done)
- Check in container (try opening a test episode)

### Tips
- Make sure the headset is on your head as the controller moves relative to it; the headset acts as the base frame.
    - Else, controller will give "senseless" values.


## To Do:
- Implement Camera Calibration: When recording demonstration high chance camera can be moved, so policy trained on uncalibrated cameras are very brittle. Best to implement camera calibration (and recording the calibration values every waypoint) to make the policy robust to accidental (or intentional) camera pose changes.
- Using tactile sensing to train diffusion policies.

## Done
- Update gripper to implement blocking stop command: https://github.com/facebookresearch/fairo/pull/1417/files
    - fork and update and then only clone that in docker
- dual camera setup
    - 2 fixed
    - hybrid: one fixed + one wrist
- record episodes
- train policy
- depoly policy
- camera setup:
    - wrist or outside?
    - multi or one?
- recording an episode
    - with or without ros
    - ee_pose (target pos or actual pos currently?) (10hz)
        - target pos makes sense
    - image stream (30hz)
    - joint angles (do i wanna)
    - open or close gripper
- update gripper to implement blocking stop command: https://github.com/facebookresearch/fairo/pull/1417/files
    - fork and update and then only clone that in docker

- not properly going down
    - tune scale values
    - remove vel safeguard
    - FIXED: had to wear headsets lol
- gripper setup: DONE
- Quest keeeps flipping turning off; prevent that howwwww?
    - Done with tape for now
- lower hz of the controller; tooooo high: helps a bit
- still jerky motion but stabilizes in a bit? : nope to stiff
- also deadband controller for controller, and ee pose and rad
    - tune values
- A button calibrate and follow vr controller, B don't -> DONE
