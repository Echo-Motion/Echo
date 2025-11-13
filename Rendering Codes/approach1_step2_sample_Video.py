import os
import glob
from tqdm import tqdm
from winniyCode.push_WeCom import text_message
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# 定义输入和输出视频目录
name = 'Methods01_Ours_Sum_model000000010'
input_video_dir = os.path.join('..','..','output','1029', f'{name}')  # 原始输入视频的目录
output_video_dir = os.path.join('..','..','output','1029', f'{name}_with_sample')  # 截取后视频的保存目录

# 如果输出目录不存在，创建该目录
if not os.path.exists(output_video_dir):
    os.makedirs(output_video_dir)

# 定义样本数据（从文本中复制）
sample_data = '''
sample01, val_2023_v0_017,    20-60,  40
sample02, val_2023_v0_008,    40-60,  20
sample03, val_2023_v0_009,    00-20,  20
sample04, val_2023_v0_038,    30-55,  25
sample05, val_2023_v0_023,    40-55,  15

sample06, val_2023_v0_025,    20-40,  20
sample07, val_2023_v0_025,    50-65,  15
sample08, val_2023_v0_033,    00-15,  15
sample09, val_2023_v0_033,    40-60,  20
sample10, val_2023_v0_031,    00-15,  15

sample11, val_2023_v0_031,    20-40,  20
sample12, val_2023_v0_029,    00-25,  25
sample13, val_2023_v0_029,    45-60,  15
sample14, val_2023_v0_039,    45-65,  20
sample15, val_2023_v0_036,    25-40,  15

sample16, val_2023_v0_001,    10-30,  20
sample17, val_2023_v0_001,    35-50,  15
sample18, val_2023_v0_010,    30-50,  20
sample19, val_2023_v0_019,    00-25,  25
sample20, val_2023_v0_021,    00-30,  30
'''

# 解析样本数据，将其转换为列表
samples = []

for line in sample_data.strip().split('\n'):
    # 跳过空行
    if not line.strip():
        continue
    # 将行拆分为字段
    fields = line.strip().split(',')
    if len(fields) != 4:
        print(f'无效的行：{line}')
        continue
    sample_id = fields[0].strip()  # 样本ID号
    file_name = fields[1].strip()  # 文件名（部分）
    clip_range = fields[2].strip()  # 截取时间范围
    duration = fields[3].strip()  # 截取时长
    # 将clip_range拆分为开始时间和结束时间
    start_time_str, end_time_str = clip_range.split('-')
    start_time = float(start_time_str)  # 截取开始时间（秒）
    end_time = float(end_time_str)  # 截取结束时间（秒）
    # 将样本数据添加到列表中
    sample = {
        'sample_id': sample_id,
        'file_name': file_name,
        'start_time': start_time,
        'end_time': end_time
    }
    samples.append(sample)

# 处理每个样本
for sample in tqdm(samples, desc='Processing samples'):
    sample_id = sample['sample_id']
    file_name = sample['file_name']
    start_time = sample['start_time']
    end_time = sample['end_time']

    # 在输入目录中查找匹配的视频文件
    # 构建匹配模式，查找以file_name开头，并以_with_audio.mp4结尾的文件
    pattern = os.path.join(input_video_dir, f"{file_name}_*_with_audio.mp4")
    print(f"查找匹配的视频文件：{pattern}")
    matching_files = glob.glob(pattern)
    if not matching_files:
        print(f"未找到匹配的视频文件：{pattern}")
        continue
    # 假设只有一个匹配的文件，取第一个
    video_file = matching_files[0]

    # 构建输出文件名，格式为 sampleID_fileName_clipTime.mp4
    # 例如：sample01_val_2023_v0_017_20-60.mp4
    clip_time_str = f"{int(start_time):02d}-{int(end_time):02d}"
    output_file_name = f"{sample_id}_{file_name}_{clip_time_str}.mp4"
    output_file = os.path.join(output_video_dir, output_file_name)

    # 提取视频片段并保存
    # 使用 moviepy 提供的 ffmpeg_extract_subclip 函数
    # 并捕获可能的异常
    try:
        ffmpeg_extract_subclip(video_file, start_time, end_time, targetname=output_file)
    except Exception as e:
        print(f"提取视频片段时发生错误：{e}")

# 全部处理完成后，发送消息通知
# text_message('视频剪辑任务完成')