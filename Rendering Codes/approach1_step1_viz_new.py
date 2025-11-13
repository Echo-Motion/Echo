# 导入所需的库
import os
import time
import pickle
import json
import requests
import datetime
import copy
import numpy as np
import matplotlib
from winniyCode.push_WeCom import text_message  # 推送消息到企业微信
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # 3D绘图工具
from sklearn.pipeline import Pipeline
from pymo.parsers import BVHParser  # 解析BVH文件
from pymo.preprocessing import *  # 预处理BVH数据
import subprocess
from pydub import AudioSegment  # 处理音频
from subprocess import Popen, PIPE

# 设置Matplotlib使用的字体，以支持中文显示
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="C:\\Windows\\Fonts\\simsun.ttc", size=12)  # 使用宋体

# 设置关节连接列表，表示关节之间的连接关系
joint_links = [
    # 身体
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    # 左腿
    (1, 7), (7, 8), (8, 9), (9, 10), (10, 11),
    # 脊柱
    (1,12), (12,13), (13,14), (14,15), (15,16),
    # 左臂
    (14,17), (17,18), (18,19), (19,20), (20,21), (21,22), (22,23), (23,24), (24,25), (25,26),
    # 右臂
    (14,48), (48,49), (49,50), (50,51), (51,52), (52,53), (53,54), (54,55), (55,56), (56,57),
    # 头部
    (14,76), (76,77),
]

# 输出关节连接的数量
print("关节连接数量：", len(joint_links))

# 定义读取BVH文件并处理BVH数据的函数
def process_bvh(bvh_filename, start=0, stop=-1):
    """
    读取BVH文件，并将其转换为需要的格式

    参数：
    - bvh_filename: BVH文件的路径
    - start: 起始帧
    - stop: 结束帧

    返回：
    - out_data: 处理后的数据，形状为 (帧数, 关节数, 3)
    """
    # 初始化BVH解析器
    p = BVHParser()

    data_all = []
    # 解析BVH文件，获取所有关节的运动数据
    data_all.append(p.parse(bvh_filename, start=start, stop=stop))

    # 创建数据处理流水线
    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=30, keep_all=False)),  # 降采样到30FPS
        ('param', MocapParameterizer('position')),  # 提取关节位置
        ('np', Numpyfier())  # 转换为numpy数组格式
    ])

    # 处理数据，通过流水线完成关节数据的提取和转换
    out_data = data_pipe.fit_transform(data_all)
    out_data = out_data[0]  # 获取处理后的数据

    return out_data

# 定义可视化函数，用于将两个角色的动作和分析结果可视化到视频中
def visualize(x, x_2, analysis_results, role1_name, role2_name, save_path=None):
    """
    可视化两个角色的动作，同时在视频中展示分析结果和说话内容

    参数：
    - x: 第一个角色的关节位置数据，形状为 (帧数, 关节数, 3)
    - x_2: 第二个角色的关节位置数据，形状为 (帧数, 关节数, 3)
    - analysis_results: 分析结果列表，与帧数同步
    - role1_name: 第一个角色的名称（例如 'mainagent'）
    - role2_name: 第二个角色的名称
    - save_path: 保存视频的路径，如果为None，则直接显示
    """
    # 创建绘图窗口
    fig = plt.figure(figsize=(10, 8))  # 调整画布大小
    ax = fig.add_subplot(1, 1, 1, projection='3d')  # 3D子图
    ax.view_init(elev=0, azim=0)  # 设置初始视角
    plt.tight_layout()

    # 初始化文本显示，位置设置在图形的上方
    text_ax = fig.add_axes([0.05, 0.85, 0.9, 0.1])  # 在图形上方添加新子图用于显示文本，调整位置和大小
    text_ax.axis('off')  # 关闭坐标轴显示

    # 获取帧数
    num_frames = len(x)

    # 预处理分析结果，将分析结果按时间戳展开到对应的帧
    frame_texts = ['' for _ in range(num_frames)]  # 每帧的要显示的文本内容

    # 遍历分析结果，提取文字和时间戳信息
    for result in analysis_results:
        if 'wordtimestamp' in result:
            word_timestamps = result['wordtimestamp']
            # 获取句子开始和结束时间（以秒为单位）
            sentence_start_time = word_timestamps[0][0]
            sentence_end_time = word_timestamps[-1][1]
            # 将时间转换为帧数
            sentence_start_frame = int(sentence_start_time * 30)
            sentence_end_frame = int(sentence_end_time * 30)

            # 确保帧数在有效范围内
            sentence_start_frame = max(0, sentence_start_frame)
            sentence_end_frame = min(num_frames - 1, sentence_end_frame)

            # 设置要显示的文本，包括句子和分析信息
            display_text = ''
            display_text += f"角色：{result.get('role', '')}\n"
            display_text += f"句子：{result.get('sentence', '')}\n"
            display_text += f"NER：{result.get('NER', [])}\n"
            display_text += f"PNER：{result.get('PNER', [])}\n"
            display_text += f"Intention：{result.get('Intention', '')}\n"
            display_text += f"CII：{result.get('CII', [])}\n"
            display_text += f"PIE：{result.get('PIE', [])}\n"
            display_text += f"Tone：{result.get('Tone', '')}\n"
            display_text += f"SLC：{result.get('SLC', '')}\n"
            display_text += f"FSSE：{result.get('FSSE', [])}\n"

            # 将文本信息分配到每一帧
            for i in range(sentence_start_frame, sentence_end_frame + 1):
                frame_texts[i] = display_text

    # 定义动画更新函数，逐帧更新显示内容
    def animate(i):
        pose = x[i]  # 第一个角色的关节位置
        pose2 = x_2[i]  # 第二个角色的关节位置
        ax.clear()  # 清除上一帧的内容

        # 绘制第一个角色的骨架
        for pair in joint_links:
            ax.plot([pose[pair[0], 0], pose[pair[1], 0]],
                    [pose[pair[0], 2], pose[pair[1], 2]],
                    [pose[pair[0], 1], pose[pair[1], 1]],
                    zdir='z', linewidth=3, color='blue')
        # 在第一个角色上方添加角色名称标签
        ax.text(pose[14, 0], pose[14, 2]-200, pose[14, 1], role1_name, color='blue', fontsize=12, fontproperties=font)

        # 绘制第二个角色的骨架，稍微平移以避免重叠
        offset = 250  # 根据实际情况调整偏移量
        for pair in joint_links:
            ax.plot([pose2[pair[0], 0]+offset, pose2[pair[1], 0]+offset],
                    [pose2[pair[0], 2], pose2[pair[1], 2]],
                    [pose2[pair[0], 1], pose2[pair[1], 1]],
                    zdir='z', linewidth=3, color='green')
        # 在第二个角色上方添加角色名称标签
        ax.text(pose2[14, 0], pose2[14, 2]+100, pose2[14, 1], role2_name, color='green', fontsize=12, fontproperties=font)

        # 设置坐标轴范围
        lim = 150
        ax.set_xlim3d(-lim, lim*2)
        ax.set_ylim3d(-lim, lim)
        ax.set_zlim3d(0, lim*2)

        # 隐藏坐标轴
        ax.axis('off')

        # 在上方文本框中显示分析结果和当前对话内容
        text_ax.clear()  # 清除上一帧的文本
        text_ax.axis('off')  # 关闭坐标轴显示

        # 获取当前帧的文字内容
        display_text = frame_texts[i]

        # 如果当前帧有文字需要显示
        if display_text:
            # 显示文本
            text_ax.text(0.5, 0.5, display_text, horizontalalignment='center', verticalalignment='center',
                         fontsize=10, color='black', transform=text_ax.transAxes, fontproperties=font)

    # 创建动画
    if save_path:
        ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=33.33, repeat=False)
        ani.save(save_path, fps=30, dpi=150)
        plt.close(fig)
    else:
        ani = animation.FuncAnimation(fig, animate, frames=num_frames, interval=33.33, repeat=False)
        plt.show()

# 裁剪音频函数，裁剪并提升音量
def trim_audio(audio_path, start_frame, stop_frame, fps=30, gain_db=10):
    """
    裁剪音频文件

    参数：
    - audio_path: 音频文件路径
    - start_frame: 起始帧
    - stop_frame: 结束帧
    - fps: 帧率，默认30
    - gain_db: 增益，默认10dB

    返回：
    - trimmed_audio: 裁剪后的音频段
    """
    start_time_ms = (start_frame / fps) * 1000  # 起始时间（毫秒）
    stop_time_ms = (stop_frame / fps) * 1000  # 结束时间（毫秒）

    audio = AudioSegment.from_wav(audio_path)  # 读取音频文件
    trimmed_audio = audio[start_time_ms:stop_time_ms]  # 裁剪音频

    trimmed_audio = trimmed_audio + gain_db  # 增加音量

    return trimmed_audio

# 合并视频和音频
def merge(mp4_path, wav_path, wav_path2):
    """
    合并视频和两个音频文件

    参数：
    - mp4_path: 视频文件路径
    - wav_path: 第一个音频文件路径
    - wav_path2: 第二个音频文件路径
    """
    out_path = mp4_path.replace('.mp4', '_with_audio.mp4')  # 输出文件路径
    cmd = ['ffmpeg', '-y', '-i', mp4_path, '-i', wav_path, '-i', wav_path2,
           '-filter_complex', '[1:a][2:a]amix=inputs=2:duration=shortest', '-strict', '-2', '-shortest', out_path]
    subprocess.call(cmd)  # 使用FFmpeg合并音频和视频

# 主函数
def run_viz(output, path, audio, path_2, audio_2, analysis_results, start, stop, role1_name, role2_name, audio_delay=10):
    """
    运行可视化，生成带有动作、声音和分析结果的视频

    参数：
    - output: 输出目录
    - path: 第一个角色的BVH文件路径
    - audio: 第一个角色的音频文件路径
    - path_2: 第二个角色的BVH文件路径
    - audio_2: 第二个角色的音频文件路径
    - analysis_results: 分析结果列表
    - start: 起始帧
    - stop: 结束帧
    - role1_name: 第一个角色的名称（例如 'mainagent'）
    - role2_name: 第二个角色的名称
    - audio_delay: 音频延迟，默认10帧
    """
    # 处理BVH文件，获取关节位置数据
    joints = process_bvh(path, start, stop + 1)
    joints2 = process_bvh(path_2, start, stop + 1)

    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)  # 创建输出目录

    # 视频文件保存路径
    mp4_path = os.path.join(output, os.path.basename(path).replace('.bvh', f'_start{start}_end{stop}.mp4'))

    # 重塑数据形状，调整为(帧数, 关节数, 3)
    x = joints.reshape((joints.shape[0], -1, 3))
    x_2 = joints2.reshape((joints2.shape[0], -1, 3))

    # 可视化并生成视频
    visualize(x, x_2, analysis_results, role1_name, role2_name, mp4_path)

    # 裁剪并调整音频
    trimmed_audio = trim_audio(audio, start + audio_delay, stop + audio_delay)
    trimmed_audio_2 = trim_audio(audio_2, start + audio_delay, stop + audio_delay)

    # 保存裁剪后的音频
    trimmed_audio_path = mp4_path.replace('.mp4', '_trimmed1.wav')
    trimmed_audio_2_path = mp4_path.replace('.mp4', '_trimmed2.wav')
    trimmed_audio.export(trimmed_audio_path, format='wav')
    trimmed_audio_2.export(trimmed_audio_2_path, format='wav')

    # 合并视频和音频
    merge(mp4_path, trimmed_audio_path, trimmed_audio_2_path)

# 读取分析结果
def read_analysis_results(result_path):
    """
    读取分析结果文件

    参数：
    - result_path: 分析结果文件的路径

    返回：
    - analysis_results: 分析结果列表
    """
    # 打开分析结果文件
    with open(result_path, 'rb') as f:
        analysis_results = pickle.load(f)

    return analysis_results

# 主程序入口
if __name__ == '__main__':
    # 设置对话编号列表
    index = [i for i in range(41)]

    for i in index:
        i_str = f'{i:03d}'  # 将编号转换为三位数的字符串

        start = 0  # 起始帧
        stop = start + 3000  # 结束帧

        # 设置角色名称
        role1_name = 'main-agent'
        role2_name = 'interloctr'

        # 设置文件路径，使用对话编号替换文件名中的占位符
        path = rf'I:\2022-2025【已投递论文资料备份】\ICMI_GENEA2023\genea2023_dataset\val\{role1_name}\bvh\val_2023_v0_{i_str}_{role1_name}.bvh'
        audio = rf'I:\2022-2025【已投递论文资料备份】\ICMI_GENEA2023\genea2023_dataset\val\{role1_name}\wav\val_2023_v0_{i_str}_{role1_name}.wav'
        path_2 = rf'I:\2022-2025【已投递论文资料备份】\ICMI_GENEA2023\genea2023_dataset\val\{role2_name}\bvh\val_2023_v0_{i_str}_{role2_name}.bvh'
        audio_2 = rf'I:\2022-2025【已投递论文资料备份】\ICMI_GENEA2023\genea2023_dataset\val\{role2_name}\wav\val_2023_v0_{i_str}_{role2_name}.wav'
        output = rf'M:\LLMDigitalHuman\codes\Pipeline_LLMs_process\val'

        # 分析结果文件路径，修改文件名格式
        result_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed',
                                   'val', 'New_LLM_k', f'New_LLM_k10', 'All_Final', f'val_2023_v0_{i_str}.pkl')

        # 读取分析结果
        analysis_results = read_analysis_results(result_path)

        # 运行可视化并生成视频
        run_viz(output, path, audio, path_2, audio_2, analysis_results, start, stop, role1_name, role2_name, audio_delay=10)

    # 完成后推送消息到企业微信
    text_message('视频生成完成！')