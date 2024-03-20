import threading
import shutil
import  os
import time

from best_face_modules.base_model.utils import  check_for_completion_flag


class FileUploaderTask(threading.Thread):

    def __init__(self,storage_path) -> None:
        super().__init__()
        self.storage_path = storage_path
        self.running = True

    def stop(self):
        self.running = False

    def zip_directory(self,path):
        is_processed = False  # 标记是否处理了任何文件夹
        # 遍历指定目录下的所有项
        for item in os.listdir(path):
            # 获取完整的文件或文件夹路径
            full_path = os.path.join(path, item)

            # 判断当前项是否为文件夹
            if os.path.isdir(full_path) and check_for_completion_flag(full_path):
                # 对文件夹进行压缩，将会创建一个zip文件
                # 第一个参数是压缩后的文件名
                # 第二个参数是格式，这里选择zip
                # 第三个参数是要压缩的目录路径
                archive_path = shutil.make_archive(full_path, 'zip', full_path)
                # 检查压缩文件是否创建成功
                if os.path.exists(archive_path):
                     # 删除原文件夹
                     shutil.rmtree(full_path)
                is_processed = True  # 处理了文件夹
        return  is_processed
    def run(self):
        while self.running:
            if not self.zip_directory(self.storage_path):
                #遍历文件夹，没有压缩文件则睡30秒
                time.sleep(30)



