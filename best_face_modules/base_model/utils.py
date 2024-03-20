import os
from best_face_modules.global_config import recording_completed_flag,image_completed_flag
def write_flag_file(output_dir,flag_name):
    # 录制完成后，在输出目录创建表示完成的标志文件
    flag_file_path = os.path.join(output_dir, flag_name)
    with open(flag_file_path, 'w') as flag_file:
        pass

def check_for_completion_flag(output_dir):
    recording_flag_file_path = os.path.join(output_dir,recording_completed_flag)
    image_flag_file_path = os.path.join(output_dir,image_completed_flag)
    return  os.path.exists(recording_flag_file_path) and os.path.exists(image_flag_file_path)







