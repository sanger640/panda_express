import numpy as np
import cv2
import pyrealsense2 as rs
import threading
import queue

class DualRealSense:
    def __init__(self, cam1_serial, cam2_serial, H, W, hz):
        # Cam 1
        self.cam1_serial = cam1_serial
        self.cam2_serial = cam2_serial
        self.H = H 
        self.W = W
        self.hz = hz
        self.p1 = rs.pipeline()
        c1 = rs.config()
        c1.enable_device(self.cam1_serial)
        c1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.hz)
        
        # Cam 2
        self.p2 = rs.pipeline()
        c2 = rs.config()
        c2.enable_device(self.cam2_serial)
        c2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, self.hz)
        
        self.p1.start(c1)
        self.p2.start(c2)
        print("Dual Cameras Started")

    def get_frames(self):
        # sync
        f1 = self.p1.wait_for_frames()
        f2 = self.p2.wait_for_frames()
        
        i1 = np.asanyarray(f1.get_color_frame().get_data())
        i2 = np.asanyarray(f2.get_color_frame().get_data())
        
        # resize and rgb
        i1 = cv2.resize(i1, (self.W, self.H))
        i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
        
        i2 = cv2.resize(i2, (self.W, self.H))
        i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
        
        # (C, H, W)
        i1 = np.transpose(i1, (2, 0, 1))
        i2 = np.transpose(i2, (2, 0, 1))
        
        return i1, i2

class DualRealsenseRecorder:
    def __init__(self, i=0, cam1_serial, cam2_serial, ssd_loc):
        self.i = i
        self.cam1_serial = cam1_serial
        self.cam2_serial = cam2_serial
        self.save_folder = ssd_loc+"episodes/" + str(self.i) + "/rgb_frames/" 
        os.makedirs(self.save_folder, exist_ok=True)
        self.running = False
        
        # Pipelines for two cameras
        self.pipeline1 = rs.pipeline()
        self.config1 = rs.config()
        self.config1.enable_device(cam1_serial)
        self.config1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline2 = rs.pipeline()
        self.config2 = rs.config()
        self.config2.enable_device(cam2_serial)
        self.config2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.frame_queue = queue.Queue(maxsize=2000)

    def start(self):
        if not self.running:
            try:
                self.pipeline1.start(self.config1)
                print(f"Camera 1 ({self.cam1_serial}) started.")
                self.pipeline2.start(self.config2)
                print(f"Camera 2 ({self.cam2_serial}) started.")
            except RuntimeError as e:
                print(f"Error starting cameras: {e}")
                return

            self.running = True
            
            # thread for capturing frame
            self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.capture_thread.start()
            
            # thread for saving to disk
            self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
            self.save_thread.start()
    
    def _capture_worker(self):
        while self.running:
            # sync frames
            frames1 = self.pipeline1.wait_for_frames()
            frames2 = self.pipeline2.wait_for_frames()
            
            color_frame1 = frames1.get_color_frame()
            color_frame2 = frames2.get_color_frame()
            
            if not color_frame1 or not color_frame2:
                continue
            
            img1 = np.asanyarray(color_frame1.get_data())
            img2 = np.asanyarray(color_frame2.get_data())
            timestamp = time.time()
            
            try:
                # add both to queue
                self.frame_queue.put_nowait((timestamp, img1, img2))
            except queue.Full:
                pass # drop if queue full (should not happen, queue big enough)

    def _save_worker(self):
        while self.running or not self.frame_queue.empty():
            try:
                timestamp, img1, img2 = self.frame_queue.get(timeout=0.1)
                
                # save cam1 and cam2 imgs with timestamp
                fname1 = os.path.join(self.save_folder, f"cam1_{int(timestamp * 1000)}.png")
                cv2.imwrite(fname1, img1)
                
                fname2 = os.path.join(self.save_folder, f"cam2_{int(timestamp * 1000)}.png")
                cv2.imwrite(fname2, img2)
                
                self.frame_queue.task_done()
            except queue.Empty:
                pass

    def stop(self):
        # stop running the recorder (close both threads)
        self.running = False
        if hasattr(self, 'capture_thread'):
            self.capture_thread.join()
        if hasattr(self, 'save_thread'):
            self.save_thread.join()
        
        try:
            self.pipeline1.stop()
            self.pipeline2.stop()
        except:
            pass
        print("Cameras stopped.")