
import os
import cv2
import torch
import numpy as np
from best_face_modules.tensorrt_detect import TRTModule  # isort:skip
from best_face_modules.tensorrt_detect.torch_utils import det_postprocess
from best_face_modules.tensorrt_detect.utils import blob, letterbox, path_to_list
import time
import threading
class TensorrtYolov8():
    def __init__(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8n_100e_int8_calibrator.engine")
        #engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8n_100e.engine")
        device_str: str = 'cuda:0'
        self.device = torch.device(device_str)
        self.Engine = TRTModule(engine_path, self.device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]
        # set desired output names order
        self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        # 现有的初始化代码...
        self.lock = threading.Lock()  # 添加这行代码来初始化锁

    def run(self,image):

        image, ratio, dwdh = letterbox(image, (self.W, self.H))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
        tensor = torch.asarray(tensor, device=self.device)
        with self.lock:  # 在这里获取锁
            # inference
            data = self.Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        if bboxes.numel() == 0:
            # if no bounding box
            return np.empty((0, 5))

        bboxes -= dwdh
        bboxes /= ratio

        bboxes_list = bboxes.tolist()
        bboxes_with_scores = [bbox + [score.cpu().item()] + [label.cpu().item()] for bbox, score, label in
                              zip(bboxes_list, scores, labels)]
        # 将 bboxes_with_scores 转换为 NumPy 数组
        dets = np.array(bboxes_with_scores)
        return  dets

    def draw_detections(self, image, boxes, scores, kpts):
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), thickness=3)
            cv2.putText(image, "face:"+str(round(score,2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 4, (0, 255, 0), thickness=-1)
        return image




if __name__ == "__main__":
        yolo = TensorrtYolov8()
        start_time = time.perf_counter()
        count = 0
        for file in os.listdir("/home/norco/soft/WIDER_FACE_YOLO/train/images/"):
            count = count + 1
            path = os.path.join("/home/norco/soft/WIDER_FACE_YOLO/train/images/", file)
            image = cv2.imread(path)
            startyolo_time = time.perf_counter()
            dets = yolo.run(image)
            yolo_time = (time.perf_counter() - startyolo_time) * 1000
            print(f"yolo_time time: {yolo_time:.3f} ms")


        track_time = (time.perf_counter() - start_time) * 1000
        print(f"total count {count}")
        print(f"total time: {track_time:.3f} ms")
        avg_time = track_time / count  # 计算平均耗时
        print(f"Average time per image: {avg_time:.3f} ms")  # 打印平均耗时

