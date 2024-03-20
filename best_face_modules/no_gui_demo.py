import cv2
import logging
import os
import sys
import queue
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#导入目录到环境变量中
from best_face_modules.face_detect_task import FaceDetectionTask
from best_face_modules.video_decode_task import VideoDecodeTask
from best_face_modules.video_decode_videos_task import VideoDecodeVideosTask
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
from best_face_modules.global_config import storage_path
from  best_face_modules.file_uploader_task import  FileUploaderTask
from line_profiler import LineProfiler
from best_face_modules.tensorrt_detect.tensorrt_detect import TensorrtYolov8
def gstreamer_pipeline(
    sensor_id=0,
    sensor_mode=3,
    capture_width=1280, #摄像头预捕获的图像宽度
    capture_height=720, #摄像头预捕获的图像高度
    display_width=1280, #窗口显示的图像宽度
    display_height=720, #窗口显示的图像高度
    framerate=21,       #捕获帧率
    flip_method=0,      #是否旋转图像
):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            sensor_mode,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def start_stream(cap, frame_queue, camera_index,face_detector):
    face_detect_task = FaceDetectionTask(cap, frame_queue, camera_index,face_detector)
    video_decode_thread = VideoDecodeTask(cap, frame_queue, face_detect_task)
    face_detect_task.start()
    video_decode_thread.start()




if __name__ == "__main__":
        face_detector = TensorrtYolov8()
        image = cv2.imread("/home/norco/soft/best_face/data/bus.jpg")
        face_detector.run(image)

        # 初始化摄像头捕获对象，使用GStreamer管道字符串
        cap1 = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
        # 创建线程安全的帧队列
        frame_queue1 = queue.Queue()
        camera_index = 0
        start_stream(cap1,frame_queue1,camera_index,face_detector)
        '''
        # 初始化摄像头捕获对象，使用GStreamer管道字符串
        cap2 = cv2.VideoCapture("/home/norco/soft/best_face/best_face_modules/output.mp4")
        # 创建线程安全的帧队列
        frame_queue2 = queue.Queue()
        camera_index2 = 1
       # start_stream(cap2, frame_queue2, camera_index2)
        face_detect_task = FaceDetectionTask(cap2, frame_queue2, camera_index2,face_detector)
        video_decode_thread = VideoDecodeVideosTask(cap2, frame_queue2, face_detect_task)
        video_decode_thread.start()
        face_detect_task.start()
        '''
        #打包上传任务
        #uploader_task = FileUploaderTask(storage_path)
        #uploader_task.start()

