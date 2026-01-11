import cv2
import numpy as np
import pyrealsense2 as rs

# --- CONFIGURATION (Must match your training YAML) ---
# 1. Input Resolution 
INPUT_W, INPUT_H = 320, 240

# 2. Crop Resolution (during aug in training)
CROP_H, CROP_W = 216, 288 
# -----------------------------------------------------

def main():
    print(f"Connecting to RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()
    
    # capture at 640x480 
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error connecting to camera: {e}")
        return

    print("\n--- VISUAL DEBUGGER RUNNING ---")
    print(f"Resizing to: {INPUT_W}x{INPUT_H}")
    print(f"Center Crop: {CROP_W}x{CROP_H}")
    print("--------------------------------")
    print("GREEN BOX = What the Neural Network sees.")
    print("OUTSIDE   = Blind spot.")
    print("--------------------------------")
    print("Press 'q' to quit.")

    try:
        while True:
            # get Frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # to numpy
            frame = np.asanyarray(color_frame.get_data())

            # resize
            debug_img = cv2.resize(frame, (INPUT_W, INPUT_H))

            # crop
            start_y = (INPUT_H - CROP_H) // 2
            start_x = (INPUT_W - CROP_W) // 2
            end_y = start_y + CROP_H
            end_x = start_x + CROP_W

            # green box
            cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
            
            # mask
            overlay = debug_img.copy()
            cv2.rectangle(overlay, (0, 0), (INPUT_W, INPUT_H), (0, 0, 0), -1) # Black out everything
            cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1) # Clear center (inverted logic for masking is harder in pure cv2, simple blend is easier)
                        

            display_img = cv2.resize(debug_img, (INPUT_W * 2, INPUT_H * 2), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Network Input Debugger", display_img)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()