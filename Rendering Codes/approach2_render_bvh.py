#!/usr/bin/env python3
import argparse
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser(description='渲染BVH动画')
    
    # 添加命令行参数
    parser.add_argument('-b', '--blender', 
                      default='D:/ProgramLoadFile/blender/blender.exe',
                      help='Blender可执行文件路径')
    
    parser.add_argument('-s', '--script',
                      default='E:/大模型绘图需求/show/change_human.py', 
                      help='渲染脚本路径')
    
    parser.add_argument('-1', '--bvh1',
                      default='M:/LLMDigitalHuman/resources/datasets/twh_genea2023_dataset/raw/trn/main-agent/bvh/trn_2023_v0_007_main-agent.bvh',
                      help='主角色BVH文件路径')
    
    parser.add_argument('-2', '--bvh2',
                      default='M:/LLMDigitalHuman/resources/datasets/twh_genea2023_dataset/raw/trn/interloctr/bvh/trn_2023_v0_007_interloctr.bvh',
                      help='交互者BVH文件路径')
    
    parser.add_argument('-w1', '--wav1',
                      default='M:/LLMDigitalHuman/resources/datasets/twh_genea2023_dataset/raw/trn/main-agent/wav/trn_2023_v0_007_main-agent.wav',
                      help='主角色WAV文件路径')
    
    parser.add_argument('-w2', '--wav2',
                      default='M:/LLMDigitalHuman/resources/datasets/twh_genea2023_dataset/raw/trn/interloctr/wav/trn_2023_v0_007_interloctr.wav',
                      help='交互者WAV文件路径')
    
    parser.add_argument('-o', '--output',
                      default='E:/test2',
                      help='输出目录')
    
    parser.add_argument('-d', '--duration',
                      type=int,
                      default=1000,
                      help='渲染帧数')
    
    parser.add_argument('-m', '--mode',
                      default='full_body',
                      help='渲染模式')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 构建命令
    cmd = [
        args.blender,
        '-b',
        '--python', args.script,
        '--',
        '-i1', args.bvh1,
        '-i2', args.bvh2,
        '-a1', args.wav1,
        '-a2', args.wav2,
        '-v',
        '-m', args.mode,
        '-o', args.output,
        '-sb',
        '-d', str(args.duration)
    ]
    
    # 执行命令
    print("执行命令:", ' '.join(cmd))
    subprocess.run(cmd)

if __name__ == '__main__':
    main()