import cv2


def on_trackbar(val):
    global cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Video', frame)


video_path = r'E:\cv\testVideo\2023-12-21_11-12-51\2023-12-21_11-12-51_167.avi'
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 设置初始帧位置
initial_frame_number = 117  # 替换为你需要的起始帧号
cap.set(cv2.CAP_PROP_POS_FRAMES, initial_frame_number)
cv2.namedWindow('Video')  # 创建窗口
cv2.createTrackbar('Position', 'Video', initial_frame_number, total_frames-1, on_trackbar)  # 创建滑动条

on_trackbar(initial_frame_number)  # 更新滑块位置并展示帧

while True:
    key = cv2.waitKey(int(1000/fps)) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
