import zmq
import time
from collections import deque
from franka import Franka
from cam import DualRealSense

# --- CONFIGURATION ---
GPU_SERVER_IP = "129.97.71.51" 
PORT = 5555
ROBOT_IP = "129.97.71.27"

# SERIAL NUMBERS (Must match training order)
CAM1_SERIAL = "215222078938" 
CAM2_SERIAL = "819612070440"


IMG_W, IMG_H = 320, 240
CONTROL_HZ = 10
# ---------------------

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{GPU_SERVER_IP}:{PORT}")
    print(f"Connected to GPU Server at {GPU_SERVER_IP}")

    cameras = DualRealSense(cam1_serial=CAM1_SERIAL, cam2_serial=CAM2_SERIAL, H=IMG_H, W=IMG_W, hz=30)
    robot = Franka(robot_ip=ROBOT_IP)


    obs_buffer = deque(maxlen=2)

    try:
        while True:
            loop_start = time.time()
            obs_buffer.clear()

            while len(obs_buffer) < 2:
                i1, i2 = cameras.get_frames()
                s = robot.get_state()
                obs_buffer.append({'c1': i1, 'c2': i2, 's': s})

            b_c1 = np.stack([x['c1'] for x in obs_buffer])
            b_c2 = np.stack([x['c2'] for x in obs_buffer])
            b_s = np.stack([x['s'] for x in obs_buffer])

            # send message to model server
            socket.send_pyobj({
                'cam1': b_c1.tobytes(),
                'cam2': b_c2.tobytes(),
                'state': b_s.tobytes(),
                'shape': (IMG_H, IMG_W)
            })

            # get actions
            response = socket.recv_pyobj()
            actions = response['action'] # (

            # execute actions
            for i, action in enumerate(actions):
                step_start = time.time()
                robot.execute(action)                
                elapsed = time.time() - step_start
                sleep_time = max(0, (1.0 / CONTROL_HZ) - elapsed)
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("stopped")
        robot.robot.terminate_current_policy()

if __name__ == "__main__":
    main()