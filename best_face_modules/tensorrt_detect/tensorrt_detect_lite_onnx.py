
import os
import cv2
import torch
import numpy as np
from best_face_modules.tensorrt_detect import TRTModule  # isort:skip
from best_face_modules.tensorrt_detect.torch_utils import det_postprocess, det_postprocess_new
from best_face_modules.tensorrt_detect.utils import blob, letterbox, path_to_list
import time
import threading
class TensorrtYolov8():
    def __init__(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        #engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8n_100e_int8_calibrator.engine")
        #engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8n_100e.engine")
        engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8-lite-t_int8_calibrator.engine")
        device_str: str = 'cuda:0'
        self.device = torch.device(device_str)
        self.Engine = TRTModule(engine_path, self.device)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]
        # set desired output names order
        # self.Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
        self.Engine.set_desired(['output0', ])
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
        print(data)
        for tensor in data:
            print(f"Shape of data: {tensor.shape}")
            print(f"Data type: {tensor.dtype}")

        data = data.cpu()
        data = data.numpy()
        bboxes, scores, labels, landmarks = det_postprocess_new(data, ratio, ratio, dwdh[1], dwdh[0])
        if len(bboxes) == 0:
            return np.empty((0, 16))

        dets = np.hstack((bboxes, scores, labels, landmarks))
        return dets

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
        image = cv2.imread("/home/norco/soft/best_face/1.jpg")
        dets = yolo.run(image)

