import cv2
import logging
import threading
import time
from best_face_modules.face_detect_task import FaceDetectionTask
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class VideoDecodeTask(threading.Thread):


    def __init__(self, capture,frame_queue,face_detect_task) -> None:
        super().__init__()
        self.capture = capture
        self.is_stopped = False
        self.frame_queue = frame_queue
        self.face_detect_task = face_detect_task

    def stop(self):
        self.is_stopped = True

    def __del__(self):
        self.stop()

    def run(self):
        start_time = time.time()  # 获取当前时间作为开始时间
        count = 0
        try:
            while self.capture.isOpened():
                # 检查是否已经运行了300秒（5分钟）
                '''
                if (time.time() - start_time) > 60:
                    print("已运行5分钟，准备退出...")
                    break
                '''
                ret, frame = self.capture.read()
                if not ret:
                    logging.error("Failed to decode frame")
                    break
                # 如果队列已满，则先丢弃一帧
                if self.frame_queue.full():
                    self.frame_queue.get()
                    logging.warning("frame_queue is full,abandon frame")
                count = count+1
                self.frame_queue.put(frame)
                length = self.frame_queue.qsize()
                print("队列的长度为:", length)


        except cv2.error as e:  # 捕获OpenCV操作的异常
            logging.error(f"OpenCV error: {e}")
        except Exception as e:  # 捕获其他意外的异常
            logging.error(f"An unexpected error occurred: {e}")
        finally:
            self.face_detect_task.stop()



