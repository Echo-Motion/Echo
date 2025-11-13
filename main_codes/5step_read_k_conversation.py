import os
import pickle
import random
from tqdm import tqdm  # 导入tqdm用于显示进度条

"""
    提供读取并样例显示功能的脚本
    用于读取之前处理过的K轮对话数据，并展示样例以便后续开发使用
    保存到 Method_01/K_Conversation/K{k}_Conversation
"""

# 设置保存路径
save_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed')
k = 10  # 关注的对话轮数

def load_processed_conversation(dataset_type, conversation_file):
    """
    加载已经处理过的对话文件

    参数:
    - dataset_type (str): 数据集类型，如 'trn', 'val', 'tst'
    - conversation_file (str): 对话文件名

    返回:
    - list: 加载的对话数据，如果失败则返回None
    """
    # 构建加载文件的路径
    processed_save_path = os.path.join(save_path, dataset_type, 'Method_01', 'K_Conversation', f'K{k}_Conversation')
    file_path = os.path.join(processed_save_path, conversation_file)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None

    # 尝试打开并加载文件
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"无法加载文件 {file_path}: {e}")
        return None

def display_sample(data, sample_size=5):
    """
    显示对话数据的样例

    参数:
    - data (list): 对话数据列表，每个元素为一个字典
    - sample_size (int): 需要展示的样例数量
    """
    if not data:
        print("没有数据可展示。")
        return

    total = len(data)
    sample_size = min(sample_size, total)
    print(f"总共有 {total} 条数据，随机展示 {sample_size} 条样例:\n")

    # 随机选择样例索引
    sample_indices = random.sample(range(total), sample_size)

    for idx in sample_indices:
        item = data[idx]
        focus = item.get('focus', {})
        history = item.get('history', [])

        print(f"样例 {idx + 1}:")
        print(f"  中心句角色: {focus.get('role', '未知')}")
        print(f"  中心句内容: {focus.get('sentence', '')}")
        print(f"  中心句时间戳: {focus.get('wordtimestamp', [])}")
        print(f"  历史对话轮数: {len(history)}")
        print("  历史对话内容:")
        for h_idx, h_item in enumerate(history):
            print(f"    {h_idx + 1}. [{h_item.get('role', '未知')}] {h_item.get('sentence', '')}")
        print("-" * 50)

def list_conversation_files(dataset_type):
    """
    列出指定数据集类型下的所有对话文件

    参数:
    - dataset_type (str): 数据集类型，如 'trn', 'val', 'tst'

    返回:
    - list: 对话文件名列表
    """
    processed_save_path = os.path.join(save_path, dataset_type, 'Method_01', 'K_Conversation', f'K{k}_Conversation')

    # 检查目录是否存在
    if not os.path.exists(processed_save_path):
        print(f"目录不存在: {processed_save_path}")
        return []

    # 获取并排序文件列表
    conversation_files = sorted(os.listdir(processed_save_path))
    return conversation_files


if __name__ == '__main__':
    import argparse

    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="读取并展示K轮对话数据的样例")
    parser.add_argument('--dataset_type', type=str, default='val', choices=['trn', 'val', 'tst'],
                        help="数据集类型，如 'trn', 'val', 'tst'")
    parser.add_argument('--sample_size', type=int, default=100,
                        help="每个对话文件展示的样例数量")
    parser.add_argument('--conversation_file', type=str, default='trn_2023_v0_000.pkl',
                        help="指定要展示的对话文件名，如果不指定则随机选择一个文件")
    args = parser.parse_args()

    dataset_type = args.dataset_type
    sample_size = args.sample_size
    specified_file = args.conversation_file

    # 获取所有对话文件
    conversation_files = list_conversation_files(dataset_type)
    if not conversation_files:
        print("没有找到任何对话文件。")
        exit(0)

    # 如果未指定文件，则随机选择一个
    if not specified_file:
        selected_file = random.choice(conversation_files)
        print(f"随机选择的对话文件: {selected_file}")
    else:
        if specified_file not in conversation_files:
            print(f"指定的文件 '{specified_file}' 不存在于数据集中。")
            print("可用的文件列表:")
            for file in conversation_files:
                print(f"  - {file}")
            exit(0)
        selected_file = specified_file

    # 加载选定的对话文件
    data = load_processed_conversation(dataset_type, selected_file)
    if data is None:
        print("无法加载对话数据。")
        exit(0)

    # 显示样例
    display_sample(data, sample_size=sample_size)