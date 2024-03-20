
import numpy as np
import cv2
from face_recognition_modules.face_detection.yolov8_face import Yolov8Face
from deep_sort_realtime.deepsort_tracker import DeepSort
yolo8face = Yolov8Face(model_path="E:/pythonproject/face_recognition_system/models/yolov8-lite-t.onnx", device="gpu")

cap = cv2.VideoCapture("E:/cv/testVideo/test.mp4")
#获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
#获取视频的宽度和高度
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#获取视频的帧数
fNUMS = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#它是一种常用的MP4视频编解码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
videoWriter = cv2.VideoWriter("D:/track/mytrack.mp4", fourcc, fps, size)

tracker = DeepSort(max_age=20)


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, 2 / 3, txt_color, thickness=1, lineType=cv2.LINE_AA)


while cap.isOpened():
    success, frame = cap.read()

    if success:
        # 将视频帧传入YOLO模型进行人脸检测
        boxes, _ = yolo8face.run(image=frame)
       # results = model(frame, conf=0.4)
        outputs = boxes

        detections = []

        if outputs is not None:
            for output in outputs:
                x1, y1, x2, y2 = list(map(int, output[:4]))
                if output[5] == 2:
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], output[4], 'car'))
                elif output[5] == 5:
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], output[4], 'bus'))
                else:
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], output[4], 'truck'))

            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_ltrb()

                box_label(frame, bbox, '#' + str(int(track_id)) + track.det_class, (167, 146, 11))

        cv2.putText(frame, "https://blog.csdn.net/zhaocj", (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("YOLOv8 Tracking", frame)
        videoWriter.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release()
videoWriter.release()
cv2.destroyAllWindows()