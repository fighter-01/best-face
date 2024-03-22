import sys
sys.path.append("/data/zhaojd-a/github_codes/best-face")
from best_face_modules.base_model.base_model import BaseModel
from best_face_modules.tracker import sort
import numpy as np
#from best_face_modules.tensorrt_detect.tensorrt_detect import TensorrtYolov8
#from best_face_modules.tensorrt_detect.infer_without_torch import TensorrtCudaYolov8
#from best_face_modules.tensorrt_detect.tensorrt_detect_lite_onnx import TensorrtYolov8
import time
class Tracker:
    def __init__(
        self,
        #detector:TensorrtYolov8,
        detector: BaseModel,
        max_age=1,
        min_hits=3,
        iou_threshold=0.3,
    ) -> None:
        self.detector = detector
        self.sort_tracker = sort.Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)


    def track(self, frame):
        # detect
        #boxes, _ = self.detector.run(image=frame)
        track_start_time = time.perf_counter()
        boxes,landmarks= self.detector.run(image=frame)
        combined_format = []

        if 0 == len(boxes):
            combined_format = np.empty((0, 16))
        else:
            for box, landmark in zip(boxes, landmarks):
                # 展开 box 前4个元素和每个 landmark 坐标点
                combined = [*box[:4]]
                for point in landmark:
                    combined.extend(point)  # 添加 landmark 中每个点的 x 和 y
                combined.extend([box[4], int(box[5])])  # 添加得分和类别
                combined_format.append(combined)

        # 打印结果查看
        # for item in combined_format:
        #     print(item)
        # 现在你可以安全地将它转换为 NumPy 数组
        tracks_array = np.array(combined_format)
        track_time = (time.perf_counter() - track_start_time) * 1000
        print(f"infer time: {track_time:.3f} ms")
        # sort
        track_start_time = time.perf_counter()
        trackers, removed_trackers = self.sort_tracker.update(tracks_array)
        track_time = (time.perf_counter() - track_start_time) * 1000
        print(f"track time: {track_time:.3f} ms")
        '''
        cascade = cv2.CascadeClassifier("/home/nvidia/soft/best_face/best_face_modules/base_model/haarcascade_frontalface_default.xml")  ## 读入分类器数据
       
         sample_image：要进行人脸检测的输入图像。
         scaleFactor：用于在图像金字塔中缩放图像的比例因子。默认值为 1.1，表示每次缩小图像的尺寸以便进行下一次检测。
         minNeighbors：指定每个候选矩形应保留的邻居矩形数。低值会导致更多的检测结果，但可能包含假阳性。高值会导致漏检，但准确率更高。
         minSize：要检测的人脸的最小尺寸。
       
        faces = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        faces_with_scores = np.hstack((faces, np.ones((faces.shape[0], 1))))
        trackers, removed_trackers = self.sort_tracker.update(faces_with_scores)
       '''


        # limit track box in frame
        for i, tracker in enumerate(trackers):
            x1, y1, x2, y2 = tracker[:4]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            trackers[i][:4] = [x1, y1, x2, y2]

        return trackers, removed_trackers


if __name__ == "__main__":
    from best_face_modules.face_detection.yolov8_face import Yolov8Face
    import cv2
    from best_face_modules.face_detection.face_landmarks import FaceLandmarks


    yolo8face = Yolov8Face(model_path="E:/pythonproject/best_face/models/yolov8-lite-t.onnx", device="gpu")
    landmarks_det = FaceLandmarks(model_path="E:/pythonproject/best_face/models/student_128.onnx", device="gpu")
    tracker = Tracker(yolo8face)
    cap = cv2.VideoCapture("E:/cv/testVideo/test.mp4")
    while True:
        ret, frame = cap.read()
        tracks, _ = tracker.track(frame)
        for track in tracks:
            bbox = track[:4]
            # get landmarks
            # landmarks = landmarks_det.run(frame, bbox)
            id = int(track[4])
            bbox = [int(i) for i in bbox]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # for landmark in landmarks:
            #     cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
