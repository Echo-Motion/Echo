import os
import pickle
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy  # 导入 copy 模块用于深度复制

# 设置并发线程数
MAX_WORKERS = 1

# 设置参数 k
k = 10

# 数据路径设置
base_path = os.path.join(
    '..', '..', 'resources', 'datasets', 'twh_genea2023_dataset',
    'processed', 'trn', 'npy_merge_text_k', f'npy_merge_text_k{k}'
)

# 各代理的输出路径
pragmatic_output_path = os.path.join(
    '..', '..', 'resources', 'datasets', 'twh_genea2023_dataset',
    'processed', 'trn', 'LLM_k', f'LLM_k{k}', 'Pragmatic_Agents'
)
semantic_output_path = os.path.join(
    '..', '..', 'resources', 'datasets', 'twh_genea2023_dataset',
    'processed', 'trn', 'LLM_k', f'LLM_k{k}', 'Semantic_Agents'
)
sentiment_output_path = os.path.join(
    '..', '..', 'resources', 'datasets', 'twh_genea2023_dataset',
    'processed', 'trn', 'LLM_k', f'LLM_k{k}', 'Sentiment_Agents'
)

# 合并后的输出路径
merge_output_path = os.path.join(
    '..', '..', 'resources', 'datasets', 'twh_genea2023_dataset',
    'processed', 'trn', 'LLM_k', f'LLM_k{k}', 'Merge'
)

# 创建合并输出文件夹，如果不存在则创建
if not os.path.exists(merge_output_path):
    os.makedirs(merge_output_path, exist_ok=True)

# 获取所有对话文件，限制为前60个
conversation_files = sorted(os.listdir(base_path))[:60]

# 收集所有任务
tasks = []
for conversation in conversation_files:
    # 读取数据
    path = os.path.join(base_path, conversation)
    with open(path, 'rb') as f:
        dialog = pickle.load(f)

    for item_index, item in enumerate(dialog):
        tasks.append((conversation, item_index, item))


# 所有错误文件
error_all = []

# 定义处理单个 item 的函数
def process_item(task):
    # 解压任务
    conversation, item_index, item = task
    conversation_name = conversation.replace('.pkl', '')

    # 构建各代理的输出文件路径
    pragmatic_folder = os.path.join(pragmatic_output_path, conversation_name, f'item{item_index + 1:03d}')
    semantic_folder = os.path.join(semantic_output_path, conversation_name, f'item{item_index + 1:03d}')
    sentiment_folder = os.path.join(sentiment_output_path, conversation_name, f'item{item_index + 1:03d}')

    pragmatic_output_file = os.path.join(pragmatic_folder, 'output.txt')
    semantic_output_file = os.path.join(semantic_folder, 'output.txt')
    sentiment_output_file = os.path.join(sentiment_folder, 'output.txt')

    # 构建合并后的输出路径
    merge_conversation_folder = os.path.join(merge_output_path, conversation_name)
    if not os.path.exists(merge_conversation_folder):
        os.makedirs(merge_conversation_folder)
    merge_item_folder = os.path.join(merge_conversation_folder, f'item{item_index + 1:03d}')


    # 检查各代理的 output.txt 是否存在
    missing_files = []
    if not os.path.exists(pragmatic_output_file):
        missing_files.append(f"Pragmatic_Agents 的输出文件 {pragmatic_output_file} 不存在。")
    if not os.path.exists(semantic_output_file):
        missing_files.append(f"Semantic_Agents 的输出文件 {semantic_output_file} 不存在。")
    if not os.path.exists(sentiment_output_file):
        missing_files.append(f"Sentiment_Agents 的输出文件 {sentiment_output_file} 不存在。")
    if missing_files:
        for msg in missing_files:
            print(msg)
        error_all.append((conversation_name, item_index + 1))
        print(f"跳过合并 {conversation_name}/item{item_index + 1:03d}.")
        return

    # 读取 Pragmatic_Agents 的 JSON 内容
    try:
        with open(pragmatic_output_file, 'r', encoding='utf-8') as f:
            pragmatic_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"读取或解析 {pragmatic_output_file} 时出错: {e}. 跳过。")
        return

    # 读取 Semantic_Agents 的 JSON 内容
    try:
        with open(semantic_output_file, 'r', encoding='utf-8') as f:
            semantic_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"读取或解析 {semantic_output_file} 时出错: {e}. 跳过。")
        return

    # 读取 Sentiment_Agents 的 JSON 内容
    try:
        with open(sentiment_output_file, 'r', encoding='utf-8') as f:
            sentiment_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"读取或解析 {sentiment_output_file} 时出错: {e}. 跳过。")
        return

    # 检查所有数据是否为列表
    if not isinstance(pragmatic_data, list):
        print(f"{pragmatic_output_file} 的内容不是列表格式。跳过。")
        return
    if not isinstance(semantic_data, list):
        print(f"{semantic_output_file} 的内容不是列表格式。跳过。")
        return
    if not isinstance(sentiment_data, list):
        print(f"{sentiment_output_file} 的内容不是列表格式。跳过。")
        return

    # 检查所有代理的数据长度是否与 item 中的对话长度一致
    if not (len(pragmatic_data) == len(semantic_data) == len(sentiment_data) == len(item)):
        print(f"{conversation_name} 的 item {item_index + 1} 中，各代理的数据长度不一致。跳过。")
        print(f"Pragmatic_Agents: {len(pragmatic_data)} 条数据")
        print(f"Semantic_Agents: {len(semantic_data)} 条数据")
        print(f"Sentiment_Agents: {len(sentiment_data)} 条数据")
        print(f"对话数据: {len(item)} 条数据")
        print("*" * 50)
        error_all.append((conversation_name, item_index + 1, len(pragmatic_data), len(semantic_data), len(sentiment_data)))

        if len(pragmatic_data) != len(item):
            os.rename(pragmatic_output_file, pragmatic_output_file.replace('.txt', '_error.txt'))
        if len(semantic_data) != len(item):
            os.rename(semantic_output_file, semantic_output_file.replace('.txt', '_error.txt'))
        if len(sentiment_data) != len(item):
            os.rename(sentiment_output_file, sentiment_output_file.replace('.txt', '_error.txt'))

        return

    if not os.path.exists(merge_item_folder):
        os.makedirs(merge_item_folder)
    merge_output_file = os.path.join(merge_item_folder, 'merged_output.json')

    # 检查是否已存在合并后的输出文件，避免重复处理
    if os.path.exists(merge_output_file):
        print(f"合并输出文件 {merge_output_file} 已存在。跳过。")
        return

    # 构建合并后的 JSON 结构
    merged_json = []
    for turn_index in range(len(item)):
        item_turn = item[turn_index]
        pragmatic_turn = pragmatic_data[turn_index]
        semantic_turn = semantic_data[turn_index]
        sentiment_turn = sentiment_data[turn_index]

        merged_entry = {
            # 从原始对话数据中获取角色和时间戳
            "role": item_turn.get('role', ''),
            "wordtimestamp": item_turn.get('wordtimestamp', ''),

            # 使用 Pragmatic_Agents 的 Text 字段，并重命名为 "Text"
            "Text": pragmatic_turn.get('Text', ''),

            # 合并各代理的 Chain-Of-Thought 字段，并添加后缀
            "Chain-Of-Thought-Pragmatic_Agents": pragmatic_turn.get('Chain-Of-Thought', ''),
            "Chain-Of-Thought-Semantic_Agents": semantic_turn.get('Chain-Of-Thought', ''),
            "Chain-Of-Thought-Sentiment_Agents": sentiment_turn.get('Chain-Of-Thought', ''),

            # 合并 Pragmatic_Agents 的其他字段
            "CII": pragmatic_turn.get('CII', []),
            "PIE": pragmatic_turn.get('PIE', []),
            "Tone": pragmatic_turn.get('Tone', ''),

            # 合并 Semantic_Agents 的其他字段
            "NER": semantic_turn.get('NER', []),
            "PNER": semantic_turn.get('PNER', []),
            "Intention": semantic_turn.get('Intention', ''),

            # 合并 Sentiment_Agents 的其他字段
            "SLC": sentiment_turn.get('SLC', ''),
            "FSSE": sentiment_turn.get('FSSE', [])
        }

        merged_json.append(merged_entry)

    # 保存合并后的 JSON 内容到合并输出文件
    try:
        with open(merge_output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_json, f, ensure_ascii=False, indent=4)
        print(f"成功合并并保存 {merge_output_file}.")
    except Exception as e:
        print(f"保存 {merge_output_file} 时出错: {e}.")


# 使用线程池并行处理任务，并显示进度条
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_item, task) for task in tasks]
    for future in tqdm(as_completed(futures), total=len(futures), desc="处理任务"):
        try:
            future.result()
        except Exception as e:
            print(f"任务执行时出错: {e}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"所有任务在 {elapsed_time:.2f} 秒内完成。")

# 打印所有错误的文件
print("以下文件处理时出现错误:")
for error in error_all:
    print(error)
print("错误文件数量:", len(error_all))