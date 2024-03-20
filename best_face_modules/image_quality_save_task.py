import threading
import queue
import cv2
import os
import logging
from best_face_modules.face_detection.face_landmarks import FaceLandmarks
from best_face_modules.base_model.utils import  write_flag_file
from best_face_modules.global_config import image_completed_flag
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
landmarks_det = FaceLandmarks(model_path="../models/student_128.onnx", device="gpu")
from best_face_modules.face_quality.face_quality import FaceQualityOverall
import numpy as np
class ImageQualitySaveTask(threading.Thread):

    def __init__(self) -> None:
        super().__init__()
        self.running = True
        self.track_info_queue = queue.Queue()
        self.tracks_dict = {}
        self.landmarks_det = landmarks_det
        self.face_quality = FaceQualityOverall()
    def stop(self):
        self.running = False

    def add_track_info(self,track_info):
        self.track_info_queue.put(track_info)

    def add_end_flag(self,path):
        self.track_info_queue.put((None, path))

    def save_images(self,tracks_dict, path):
        for track_id, track_info in tracks_dict.items():
            fileName = os.path.join(path, str(track_id) + "_" + str(track_info.location) + "_" + str(
                track_info.key_frames[0].quality))
            try:
                cv2.imwrite(fileName + '.jpg', track_info.key_frames[0].image)
            except cv2.error as e:
                logging.error(f"Failed to save image: {e}")
            except Exception as e:
                logging.error(f"An unexpected error occurred when saving image: {e}")

    '''
     根据传入的track_info跟踪信息，以及tracks_dict跟踪信息字典，来跟新字段中的track_info信息
     逻辑：对传入的单个track_info进行质量检测，如果分值大于字典中已存在的分值，则跟新，分值不做任何操作
     如果字典中没有track_info对应的id，则直接添加
    '''
    def quality_worker(self, track_info, tracks_dict):

        track_id = track_info.id
        image = track_info.key_frames[0].image
        face_box = track_info.bbox
        # 获取关键点
        #landmarks = self.landmarks_det.run(image, face_box)
        landmarks = np.array([
            [111.45785, 95.98847],
            [140.49316, 97.46153],
            [134.5846, 123.02166],
            [107.2427, 137.94365],
            [127.56242, 137.94661]
        ])

        # 计算分值
        quality = self.face_quality.run(image, face_box, landmarks)

        track_info.key_frames[0].quality = quality
        track_info.key_frames[0].landmarks = landmarks
        if track_info.id in tracks_dict.keys():
            if tracks_dict[track_id].key_frames[0].quality < quality:
                tracks_dict[track_id] = track_info
        else:
            tracks_dict[track_id] = track_info

    def run(self):
        while self.running or not self.track_info_queue.empty():
            try:
                print("face_landmark:" + str(self.track_info_queue.qsize()))
                item  = self.track_info_queue.get(timeout=30)

            except Exception :
                continue
            # 检查是否是结束标识
            if isinstance(item, tuple) and item[0] is None:

                self.save_images(self.tracks_dict, item[1])
                # 保存完图片后，写入标示文件
                write_flag_file(item[1], image_completed_flag)
                self.tracks_dict={}
            else:
                self.quality_worker(item,self.tracks_dict)




