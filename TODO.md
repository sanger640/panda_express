## To DO
- Launch Gripper and Robot URDF
- Add singularity safety measures
- Much slower speed
- No VR Headset fix (add tape to prox sensor)



DOCKER_BUILDKIT=0 docker compose up
conda init
conda activate polymetis-local
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.readonly=true ip=129.97.71.27
launch_robot.py robot_client=franka_hardware ip=129.97.71.27
launch_gripper.py ip=129.97.71.27

pkill -9 run_server

python host_server.py

in franka_hardware.yaml and franka_hand.yaml: change the robot_ip to : "129.97.71.19"

https://129.97.71.84:8443/quest_controller.html



Robohub pw: Bu$hR@id

Make sure the headset is at head height, else wont work (why?)

- update gripper to implement blocking stop command: https://github.com/facebookresearch/fairo/pull/1417/files
    - fork and update and then only clone that in docker
## To Do:
- update gripper to implement blocking stop command: https://github.com/facebookresearch/fairo/pull/1417/files
    - fork and update and then only clone that in docker

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
- record 50
- preprocess data
- train policy
- deploy (on polymetis)


- tuning pid if u wanna, and also deadband, works super well rn (optional)
- prox sensor better fix (optional)
## Done
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