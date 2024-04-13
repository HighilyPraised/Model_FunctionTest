from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# 输入和输出视频文件路径
input_video_path = "./vedios/nature_walk_1920x720.mp4"
output_video_path = "./vedios/output.mp4"

# 我们想要的视频长度（秒）
desired_video_length = 8

# 提取子剪辑
ffmpeg_extract_subclip(input_video_path, 0, desired_video_length, targetname=output_video_path)