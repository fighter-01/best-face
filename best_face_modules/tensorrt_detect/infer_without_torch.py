import os
import cv2
import numpy as np
from best_face_modules.tensorrt_detect.utils import blob, det_postprocess, letterbox, path_to_list
from best_face_modules.tensorrt_detect.pycuda_api import TRTEngine
import time
class TensorrtCudaYolov8():

    def __init__(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8n_100e.engine")
        self.Engine = TRTEngine(engine_path)
        self.H, self.W = self.Engine.inp_info[0].shape[-2:]


    def run(self, image):
        image, ratio, dwdh = letterbox(image, (self.W, self.H))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)

        dwdh = np.array(dwdh * 2, dtype=np.float32)
        tensor = np.ascontiguousarray(tensor)

        # inference
        data = self.Engine(tensor)

        bboxes, scores, labels = det_postprocess(data)
        if bboxes.size  == 0:
            # if no bounding box
            return np.empty((0, 5))

        bboxes -= dwdh
        bboxes /= ratio

        bboxes_list = bboxes.tolist()
        bboxes_with_scores = [bbox + [score.item()] + [label.item()] for bbox, score, label in zip(bboxes, scores, labels)]

        # 将 bboxes_with_scores 转换为 NumPy 数组
        dets = np.array(bboxes_with_scores)
        return dets
if __name__ == "__main__":
    yolo = TensorrtCudaYolov8()
    start_time = time.perf_counter()
    count = 0
    for file in os.listdir("/home/norco/soft/WIDER_FACE_YOLO/train/images/"):
        count = count+1
        path = os.path.join("/home/norco/soft/WIDER_FACE_YOLO/train/images/", file)
        image = cv2.imread(path)
        startyolo_time = time.perf_counter()
        dets = yolo.run(image)
        yolo_time = (time.perf_counter() - startyolo_time) * 1000
        print(f"yolo_time time: {yolo_time:.3f} ms")
        #print(dets)


    track_time = (time.perf_counter() - start_time) * 1000
    print(f"total count {count}")
    print(f"total time: {track_time:.3f} ms")
    avg_time = track_time / count  # 计算平均耗时
    print(f"Average time per image: {avg_time:.3f} ms")  # 打印平均耗时


