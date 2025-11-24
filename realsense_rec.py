import pyrealsense2 as rs
import numpy as np
import cv2
import time

def record_rgb(duration_sec=10, width=320, height=240, fps=30, output_file="output.avi"):
    # Configure pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # Start streaming
    pipeline.start(config)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    start_time = time.time()

    try:
        while True:
            # Wait for a coherent frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Write frame to video file
            out.write(color_image)

            # Optional: show the stream (comment out if running headless)
            cv2.imshow('RGB Stream', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Stop after duration_sec
            if time.time() - start_time > duration_sec:
                break
    finally:
        # Stop streaming
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    record_rgb()
