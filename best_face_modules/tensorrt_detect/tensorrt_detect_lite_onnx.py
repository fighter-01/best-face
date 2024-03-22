
import os
import cv2
import torch
import numpy as np
from best_face_modules.tensorrt_detect import TRTModule  # isort:skip
from best_face_modules.tensorrt_detect.torch_utils import det_postprocess, det_postprocess_new
from best_face_modules.tensorrt_detect.utils import blob, letterbox, path_to_list
import time
from datetime import datetime
import threading
class TensorrtYolov8():
    def __init__(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        #engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8n_100e_int8_calibrator.engine")
        #engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8n_100e.engine")
        #engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8-lite-t-my_int8_calibrator.engine")
        engine_path = os.path.join(script_dir, "..", "..", "models", "yolov8-lite-t-fp16.engine")
        print(engine_path)
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
        # 在进行任何处理前复制原始图像
        original_image = image.copy()
        image, ratio, dwdh = letterbox(image, (self.W, self.H))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.device)
        tensor = torch.asarray(tensor, device=self.device)
        with self.lock:  # 在这里获取锁
            # inference
            data = self.Engine(tensor)
        '''
        print(data)
        print("ratio:"+str(ratio))
        print("dwdh[1].cpu():"+str(dwdh[1].cpu()))
        print( "dwdh[0].cpu():"+str(dwdh[0].cpu()))
        for tensor in data:
            print(f"Shape of data: {tensor.shape}")
            print(f"Data type: {tensor.dtype}")
        print(data.cpu().numpy())
        '''
        #bboxes, scores, labels, landmarks = det_postprocess_new(data[0].cpu().numpy(), ratio, ratio, dwdh[1], dwdh[0])
        #bboxes, scores, labels, landmarks = det_postprocess_new(data.cpu().numpy(), ratio, ratio, dwdh[1].cpu(), dwdh[0].cpu())
        data = data.cpu()
        data = data.numpy()
        bboxes, scores, labels, landmarks = det_postprocess_new(data, 1/ratio, 1/ratio, dwdh[1].cpu(), dwdh[0].cpu())
        if len(bboxes) == 0:
            return np.empty((0, 16))
        '''
        dstimg = draw_detections(self, original_image, bboxes, scores, landmarks)
        winName = 'Deep learning face detection use OpenCV'
        cv2.namedWindow(winName, 0)
        cv2.imshow(winName, dstimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 使用当前时间作为文件名的一部分
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'result_{current_time}.jpg'
        cv2.imwrite(filename, dstimg)
       '''
        # 将bboxes中的值转化为[x1,y1,x2,y2]
        bboxes = [[x, y, x + w, y + h] for x, y, w, h in bboxes]
        # 将 scores 转换为二维数组，使其形状变为 [2, 1]
        scores = scores[:, np.newaxis]
        dets = np.hstack((bboxes, scores))
        return dets

def draw_detections(self, image, boxes, scores, landmarks):
        for box, score, kp in zip(boxes, scores, landmarks):
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
        image = cv2.imread("/home/norco/soft/best_face/best_face_modules/tensorrt_detect/1_Handshaking_Handshaking_1_46.jpg")
        bboxes, scores, labels, landmarks = yolo.run(image)
        dstimg =  yolo.draw_detections(image,bboxes,scores,landmarks)
        cv2.imwrite('result.jpg', dstimg)
'''
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
'''
