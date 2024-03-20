import cv2

# 定义一个构建GStreamer管道字符串的函数
# 设置gstreamer管道参数
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

# 初始化摄像头捕获对象，使用GStreamer管道字符串
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("CSI摄像头初始化失败")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("获取帧失败")
        break

    cv2.imshow('CSI Camera', frame)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
