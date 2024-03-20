storage_path = r"E:\cv\testVideo" #录制视频、抓取人脸存储位置
max_duration = 60 * 30    #最大录制时长
max_without_faces = 5    #没检测到人脸的最大帧数，达到后停止录制
#track settings
max_age_config = 20   #追踪对象可以保持多少帧没有被再次检测到而不被丢弃
min_hits_config = 5  #表示一个对象需要在多少帧内被检测到才被视为有效追踪目标
image_completed_flag = "image_completed.flag"
recording_completed_flag ="recording_completed.flag"



model_repo_path="" # Path to the model repository
registery_db = "registry.sqlite" # Path to the database file
record_db = "record.sqlite" # Path to the database file
recognition_threshold = 0.5 # Threshold for face recognition
add_person_threshold = 0.6 # Threshold for adding a new person
upload_hits = 50 # Number of frames to wait before uploading a face to the server
quality_threshold = 0.5 # Threshold for face quality
register_unkonwn = True # Register unknown faces automatically    

#crop settings
crop_width_margin = 3
crop_height_margin = 3


