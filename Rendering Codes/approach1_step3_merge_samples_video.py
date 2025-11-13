# 导入必要的模块
import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from winniyCode.push_WeCom import text_message  # 导入发送微信消息的模块

# 定义最大并发请求数
MAX_Requests = 1


def merge_videos(video_name, input_dirs, labels, output_dir, font_path):
    """
    合并四个视频文件为一个横向拼接的视频，并添加文本叠加。

    参数:
    - video_name: 要处理的视频文件名
    - input_dirs: 包含四个视频的输入文件夹路径列表
    - labels: 每个输入文件夹对应的标签列表
    - output_dir: 输出视频文件夹路径
    - font_path: 字体文件的路径
    """
    try:
        # 获取每个视频的完整路径
        input_videos = [os.path.join(dir_path, video_name) for dir_path in input_dirs]

        # 检查所有输入视频是否存在
        for video in input_videos:
            if not os.path.isfile(video):
                print(f"警告: 视频文件 {video} 不存在，跳过此文件。")
                return

        # 定义输出视频的完整路径
        output_video = os.path.join(output_dir, video_name)

        # 构建带有文本叠加和统一尺寸的 ffmpeg filter_complex
        # 为每个视频添加文字叠加，并统一高度为720p（可根据实际情况调整）
        filter_complex = ""
        for i in range(len(input_videos)):
            filter_complex += (
                f"[{i}:v]scale=-1:720,drawtext=fontfile='{font_path}':text='{labels[i]}':"
                f"x=(w-text_w)/2:y=10:fontsize=24:fontcolor=black,format=yuv420p[v{i}];"
            )

        # 拼接所有处理后的视频
        filter_complex += "".join([f"[v{i}]" for i in range(len(input_videos))])
        filter_complex += f"hstack=inputs={len(input_videos)}[v]"
        # filter_complex += "[v0][v1][v2][v3][v4]hstack=inputs=5[v]"

        # 构建 ffmpeg 命令，并选择第一个视频的音频
        ffmpeg_command = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            *(['-i', video] for video in input_videos),
            '-filter_complex', filter_complex,
            '-map', '[v]',  # 使用拼接后的视频流
            '-map', '0:a?',  # 选择第一个视频的音频，如果存在
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'veryfast',
            output_video
        ]

        # 扁平化ffmpeg_command列表
        ffmpeg_command = [item for sublist in ffmpeg_command for item in
                          (sublist if isinstance(sublist, list) else [sublist])]

        # 运行 ffmpeg 命令，并捕获输出
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 检查ffmpeg执行是否成功
        if result.returncode != 0:
            print(f"错误: 处理视频 {video_name} 时发生错误。错误信息:\n{result.stderr}")
        else:
            print(f"成功: 视频 {video_name} 已成功合并。")

    except Exception as e:
        print(f"错误: 处理视频 {video_name} 时发生未预期的错误。错误信息: {e}")


def main():
    # 定义当前工作目录
    current_dir = os.path.join('..', '..', 'output', '1029')

    # 定义四个输入文件夹的名称和标签
    input_folders = [
        ('Co_sample_Co_model000050000_with_sample', 'Co'),
        ('Diff_sample_Diff_model000050000_with_sample', 'Diff'),
        ('Diff2_sample_Diff_model000050000_with_sample', 'Diff2'),
        ('DSG_sample_DSG_model000050000_with_sample', 'DSG'),
        ('Methods01_Ours_Sum_model000000010_with_sample', 'Sum'),
        ('val_with_sample', 'val')
    ]

    # 使用 os.path.join 构建每个文件夹的完整路径
    input_dirs = [os.path.join(current_dir, folder[0]) for folder in input_folders]
    labels = [folder[1] for folder in input_folders]

    # 定义输出文件夹路径，并创建该文件夹如果它不存在
    output_dir = os.path.join(current_dir, 'Z2_Compared_All_Samples')
    os.makedirs(output_dir, exist_ok=True)

    # 定义系统默认字体的路径（Windows系统）
    # 使用双反斜杠或正斜杠，确保ffmpeg能够正确识别路径
    font_path = os.path.join('C:', os.sep, 'Windows', 'Fonts', 'arial.ttf').replace("\\", "/")

    # 检查字体文件是否存在
    if not os.path.isfile(font_path):
        print(f"错误: 字体文件 {font_path} 不存在。请检查字体路径。")
        return

    # 获取 val_with_sample 文件夹中的所有视频文件名
    val_sample_dir = input_dirs[3]  # 第四个文件夹
    all_video_files = sorted([
        f for f in os.listdir(val_sample_dir)
        if os.path.isfile(os.path.join(val_sample_dir, f)) and f.lower().endswith('.mp4')
    ])

    if not all_video_files:
        print("错误: 在 val_with_sample 文件夹中未找到任何视频文件。")
        return

    # 使用 ThreadPoolExecutor 进行多线程处理
    with ThreadPoolExecutor(max_workers=MAX_Requests) as executor:
        # 提交所有任务到线程池
        futures = [
            executor.submit(merge_videos, video, input_dirs, labels, output_dir, font_path)
            for video in all_video_files
        ]

        # 使用 tqdm 显示进度条
        for _ in tqdm(as_completed(futures), total=len(futures), desc="合并视频进度"):
            pass  # tqdm 会自动更新进度条

    # 发送微信消息通知任务完成
    text_message('所有视频合并任务完成')


if __name__ == "__main__":
    main()