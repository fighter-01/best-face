import cv2
import time
import logging
import os
from datetime import datetime
import threading
#from best_face_modules.tensorrt_detect.tensorrt_detect import TensorrtYolov8
from line_profiler import LineProfiler
from best_face_modules.video_record_task import VideoRecordTask
from best_face_modules.image_quality_save_task import ImageQualitySaveTask
from best_face_modules.tracker.tracker import Tracker
from best_face_modules.base_model.common import crop_face_track, RecognizeRecord, TrackInfo
from best_face_modules.global_config import storage_path, max_duration, max_without_faces, max_age_config, \
    min_hits_config,image_completed_flag

#face_detector = Yolov8Face(model_path="../models/yolov8-lite-t.onnx", device="gpu")
#face_detector = TensorrtCudaYolov8()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



class FaceDetectionTask(threading.Thread):
    def __init__(self, capture, frame_queue,camera_index,face_detector) -> None:
        super().__init__()
        self.frame_queue = frame_queue
        self.capture = capture
        self.running = True
        self.storage_path = os.path.join(storage_path,str(camera_index))
        self.video_record_thread = VideoRecordTask(self.capture)
        self.video_record_thread.start()
        self.camera_index = str(camera_index)
        self.image_quality_save_thread =  ImageQualitySaveTask()
        self.image_quality_save_thread.start()
       # min_hits：开始报告跟踪目标之前它必须被检测到的最小帧数。
        self.face_tracker = Tracker(face_detector, max_age=max_age_config, min_hits=min_hits_config)
    def stop(self):
            self.running = False


    # 比对字典中的图片，并去重
    def recognition_dedupe(self, embedding_dict, tracks_dict):
        for track_id, embedding in embedding_dict.items():
            if track_id not in tracks_dict:
                continue
            face_ids, scores = self.face_search.search(embedding, 10)
            quality = tracks_dict[track_id].key_frames[0].quality
            aggTrack_id = ""
            for person_id, score in zip(face_ids, scores):

                if person_id not in tracks_dict:
                    continue
                if track_id == person_id:
                    continue
                if score > 0.4:
                    aggTrack_id = str(aggTrack_id) + "_" + str(person_id)
                    if quality >= tracks_dict[person_id].key_frames[0].quality:
                        tracks_dict.pop(person_id, None)
                    else:
                        tracks_dict.pop(track_id, None)
                        quality = tracks_dict[person_id].key_frames[0].quality
                        track_id = person_id



    # 所有的图片添加到特征向量库中，用于比对
    def add_embedding(self, tracks_dict, embedding_dict):
        for track_id, track_info in tracks_dict.items():
            # fileName= str(track_id)+"_"+str(track_info.key_frames[0].quality)
            # cv2.imwrite('E:/c++/videopipe/opencv/genImga/'+fileName+'.jpg', track_info.key_frames[0].image)
            embedding = self.face_recognition.run(track_info.key_frames[0].image, track_info.key_frames[0].landmarks)
            self.face_search.add(embedding, track_id)
            embedding_dict[track_id] = embedding

    def draw_show_frame(self, tracks, frame):
        # draw tracks
        # 在帧上画框，用于展示
        for track in tracks:
            x1, y1, x2, y2 = track[:4]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # chinese_text_drawer.draw_text(frame, (int(x1), int(y1)), str(track[4]), 20, (0, 255, 0))

        # 等比例缩放到960x540，放到背景图上，居中
        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False  # 返回一个标志值表示需要退出主循环
        return True

    def run(self):

        recording = False
        start_time = None
        segment_start_frame = None  # 录制视频段开始时的帧计数器
        # 未检测到人脸计数器
        frame_count_without_faces = 0
        video_image_path = self.storage_path
        frame_skip_interval = 1  # 定义跳帧间隔，例如每处理1帧，跳过4帧，则此值为5
        try:
            count = 0
            while  self.running or  not self.frame_queue.empty():
                frame_start_time = time.perf_counter()
                count += 1
                should_process_frame = count % frame_skip_interval == 0  # 判断是否应该处理当前帧
                try:
                    frame = self.frame_queue.get(timeout=30)
                    print("face_deteck"+self.camera_index+":"+str(self.frame_queue.qsize()))
                except Exception :
                    continue
                if should_process_frame:
                    track_start_time = time.perf_counter()
                    tracks, rm_track_ids = self.face_tracker.track(frame)
                    track_time = (time.perf_counter() - track_start_time) * 1000
                    print(f"total time{self.camera_index}: {track_time:.3f} ms")
                else: tracks=[]
                one_time = (time.perf_counter() - frame_start_time) * 1000
                print(f"one time{self.camera_index}: {one_time:.3f} ms")
                # 如果检测到人脸而且之前没有在录制，则初始化VideoWriter
                if len(tracks) > 0 and not recording:
                    start_time = time.time()
                    # 将日期时间对象转换为字符串
                    current_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    # 组合出要创建的目录路径
                    video_image_path = os.path.join(self.storage_path,current_time_str)
                    # 录制前创建存储目录（递归创建）
                    os.makedirs(video_image_path, exist_ok=True)
                    # 初始化录制视频对象
                    video_name = os.path.join(video_image_path, current_time_str + "_" + str(count) + '.avi')
                    recording = True
                    segment_start_frame = count  # 记录视频段开始的帧计数器
                    frame_count_without_faces = 0  # 检测到人脸时重置该计数器
                # 如果未检测到人脸且正在录制，则增加无人脸帧计数器
                if len(tracks) == 0 and recording:
                    frame_count_without_faces += 1
                # 假如有人脸则把计数器置0
                if len(tracks) > 0 and recording:
                    frame_count_without_faces = 0
                if recording:
                    # 将要录制的帧加入到队列中
                    self.video_record_thread.add_frame(video_name, frame)
                    # 使用相对于录制开始的局部帧计数和时间戳
                    local_frame_count = count - segment_start_frame
                    elapsed_time = time.time() - start_time
                    #停止录制
                    if frame_count_without_faces > max_without_faces or elapsed_time > max_duration:
                        recording = False
                        start_time = None
                        # 在队列中加入结束标志
                        self.video_record_thread.add_frame(None, None)
                        self.image_quality_save_thread.add_end_flag(video_image_path)
                tow_time = (time.perf_counter() - frame_start_time) * 1000
                print(f"tow time{self.camera_index}: {tow_time:.3f} ms")
                for track in tracks:
                    cropped_track = crop_face_track(frame, track)
                    cropped_track.location = local_frame_count
                    #将trackInfo加入到处理队列中
                    self.image_quality_save_thread.add_track_info(cropped_track)
                    del cropped_track  # 及时释放不再需要的对象
                # imageName = f"{cropped_track.id}_{local_frame_count}_{cropped_track.key_frames[0].quality}"
                # cv2.imwrite('E:/c++/videopipe/opencv/genImga/' + imageName + '.jpg', cropped_track.key_frames[0].image)
                # 显示帧
                frame_end_time = time.perf_counter()
                processing_time = (frame_end_time - frame_start_time) * 1000  # 计算这一帧的处理时间
                print(f"Elapsed time in milliseconds: {processing_time:.3f} ms")

                #should_continue = self.draw_show_frame(tracks, frame)
                #if not should_continue:  # 检查绘图函数的返回值
                #   break  # 如果返回False，则退出while循环
        except cv2.error as e:  # 捕获OpenCV操作的异常
            logging.error(f"OpenCV error: {e}")
            logging.error("An error occurred", exc_info=True)
        except Exception as e:  # 捕获其他意外的异常
            logging.error(f"An unexpected error occurred: {e}")
            logging.error("An error occurred", exc_info=True)
        finally:
            logging.error(f"停止线程")
            self.video_record_thread.stop()
            self.video_record_thread.join()

            self.image_quality_save_thread.stop()
            self.image_quality_save_thread.join()
            if self.capture:
                self.capture.release()
                self.capture = None
            # 添加特征向量到向量库，并添加到向量字典用于循环比对
            # self.add_embedding(tracks_dict, embedding_dict)
            # 通过比对，去重
            # self.recognition_dedupe(embedding_dict, tracks_dict)


