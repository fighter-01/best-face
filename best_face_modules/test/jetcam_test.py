
import cv2
import logging
from jetcam.csi_camera import CSICamera
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



camera = CSICamera(capture_device=0, width=1080, height=720, capture_width=1080, capture_height=720, capture_fps=21)
image = camera.read()  # BGR8
print(image.shape)
print(camera.value.shape)
camera.running = True  # 开始摄像头视频流

try:
    while True:
        frame = camera.value  # 获取当前帧的图像

        # 显示图像，窗口标题为'Camera'
        cv2.imshow('Camera', frame)

        # 检测按键是否按下，如果是 'q'，则退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:  # 允许通过Ctrl+C来中断程序
    pass
except Exception as e :
    logging.error(f"读取视频流异常: {e}")

finally:
    camera.running = False  # 停止摄像头视频流
