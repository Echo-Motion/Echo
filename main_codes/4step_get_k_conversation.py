import os
import pickle
from tqdm import tqdm  # 导入tqdm用于显示进度条

"""
    根据合并的对话数据切割多轮对话
    保存每个句子的之前K轮对话信息
    保存到 Method_01/K_Conversation/K{k}_Conversation
"""

# 设置保存路径
save_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed')
k = 10  # 关注的对话轮数


def read_conversation(dataset_type):
    """
    读取并处理对话数据，根据每个句子前K轮的对话生成新的数据结构并保存。

    参数:
    - dataset_type (str): 数据集类型，如 'trn', 'val', 'tst'
    """
    # 获取文本对话数据的根目录
    text_save_path = os.path.join(save_path, dataset_type, 'npy_merge_text')

    # 获取对话文件列表并按名称排序
    try:
        conversation_files = sorted(os.listdir(text_save_path))
    except FileNotFoundError:
        print(f"路径不存在: {text_save_path}")
        return

    # 设置新的保存路径
    merge_save_path = os.path.join(save_path, dataset_type, 'Method_01', 'K_Conversation', f'K{k}_Conversation')

    # 如果保存路径不存在，则创建
    if not os.path.exists(merge_save_path):
        os.makedirs(merge_save_path)
        print(f"已创建目录: {merge_save_path}")

    # 使用tqdm显示对话文件处理进度
    for i, conversation in enumerate(tqdm(conversation_files, desc=f"处理{dataset_type}数据集中的对话文件")):
        # 构建每个对话文件的完整路径
        path = os.path.join(text_save_path, conversation)

        # 读取对话数据
        try:
            with open(path, 'rb') as f:
                dialog = pickle.load(f)
        except Exception as e:
            print(f"无法读取文件 {path}: {e}")
            continue

        """ 读取结果示例
        dialog: [
            {'role': 'interloctr', 'sentence': 'Yeah. Mm', 'wordtimestamp': [[0.0, 0.04, 'Yeah.'], [6.14, 6.37, 'Mm']]},
            {'role': 'mainagent', 'sentence': 'Yeah', 'wordtimestamp': [[23.74, 24.24, 'Yeah']]},
            ...
        ]
        """

        # 初始化存储新数据项的列表
        new_items = []

        # 获取对话的长度
        dialog_length = len(dialog)

        # 遍历对话中的每个句子，作为中心句
        for idx in range(dialog_length):
            # 计算历史窗口的起始索引
            start_idx = max(0, idx - k)  # 确保起始索引不小于0

            # 当前句子的历史对话内容（前K轮）
            history = dialog[start_idx:idx]  # 不包括当前句子

            # 提取当前的中心句
            focus = dialog[idx]

            # 构建包含'focus'和'history'的新数据项
            item = {
                'focus_start_frame': focus['wordtimestamp'][0][0],  # 中心句的起始时间戳
                'focus_end_frame': focus['wordtimestamp'][-1][1],  # 中心句的结束时间戳
                'focus_word_time_stamp': focus['wordtimestamp'],  # 中心句的每个词的时间戳
                'focus': focus,  # 中心句的完整信息
                'history': history  # 中心句之前K轮的对话历史
            }

            # 将新数据项添加到列表中
            new_items.append(item)

        # 保存处理后的新数据到文件
        save_file_path = os.path.join(merge_save_path, conversation)
        try:
            with open(save_file_path, 'wb') as f:
                pickle.dump(new_items, f)
        except Exception as e:
            print(f"无法保存文件 {save_file_path}: {e}")
            continue

        print(f"已处理对话 {i + 1}/{len(conversation_files)}: {conversation}")


if __name__ == '__main__':
    # 定义需要处理的数据集类型，可以根据需要添加 'tst' 等其他类型
    dataset_types = ['trn', 'val']

    for dataset_type in dataset_types:
        print('*' * 50, f"开始处理数据集: {dataset_type}", "*" * 50)
        read_conversation(dataset_type)
        print(f"完成处理数据集: {dataset_type}\n")