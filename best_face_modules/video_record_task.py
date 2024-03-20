import threading
import cv2
import os
import queue
from best_face_modules.base_model.utils import  write_flag_file
from best_face_modules.global_config import recording_completed_flag
class VideoRecordTask(threading.Thread):

    def __init__(self,capture) -> None:
        super().__init__()
        self.running = True
        self.record_queue = queue.Queue()
        self.video_writer = None
        self.capture = capture
        self.current_video_name = None
    def stop(self):
        self.running = False


    def add_frame(self,video_name,frame):
        if not self.running :
            return
        self.record_queue.put((video_name,frame))

     # 初始化视频录制对象
    def init_record_video(self,videoName):
        # 获取摄像头实际的帧率
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        # 宽度和高度也可以从摄像头获取，确保录制视频的宽高与摄像头一致
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # MPEG - 4编解码器，通常有很好的压缩率和兼容性
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(videoName, fourcc, fps, (width, height))
        return out

    def release_resources(self):
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
    def run(self):
        while self.running or not self.record_queue.empty():
            try:
                print("video_record:" + str(self.record_queue.qsize()))
                video_name, frame = self.record_queue.get(timeout=30)

            except Exception:
                continue
            if self.video_writer is None:
                self.video_writer = self.init_record_video(video_name)
            if  video_name is None:
                self.release_resources()
                write_flag_file(os.path.dirname( self.current_video_name),recording_completed_flag)
                continue
            self.video_writer.write(frame)
            self.current_video_name =video_name

        self.release_resources()
        write_flag_file(os.path.dirname(self.current_video_name), recording_completed_flag)