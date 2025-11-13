import os
import json
import pickle
from tqdm import tqdm

k = 10
output_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed', 'trn', 'LLM_k', f'LLM_k{k}', 'Sentiment_Agents')
conversation_folders = sorted(os.listdir(output_path))

# 用于记录没有正确获得输出结果的 items
error_items = []

for conversation in tqdm(conversation_folders):
    conversation_folder = os.path.join(output_path, conversation)
    item_folders = sorted(os.listdir(conversation_folder))
    for item_folder in item_folders:
        item_folder_path = os.path.join(conversation_folder, item_folder)
        input_file = os.path.join(item_folder_path, 'input.txt')
        output_file = os.path.join(item_folder_path, 'output.txt')

        # 检查 output.txt 是否存在
        if not os.path.exists(output_file):
            print(f"[Error] Missing output file for Conversation: {conversation}, Item: {item_folder}")
            error_items.append((conversation, item_folder))
            continue  # 跳过当前 item

        try:
            # 读取输入和输出 JSON 数据
            with open(input_file, 'r', encoding='utf-8') as f:
                input_content = json.load(f)

            with open(output_file, 'r', encoding='utf-8') as f:
                output_content = json.load(f)

            # pickle 保存
            with open(os.path.join(item_folder_path, 'input.pkl'), 'wb') as f:
                pickle.dump(input_content, f)

            with open(os.path.join(item_folder_path, 'output.pkl'), 'wb') as f:
                pickle.dump(output_content, f)

            # pickle 读取
            with open(os.path.join(item_folder_path, 'input.pkl'), 'rb') as f:
                input_content = pickle.load(f)

            with open(os.path.join(item_folder_path, 'output.pkl'), 'rb') as f:
                output_content = pickle.load(f)


            # 在此处可以进行快速处理
            # print(f"Conversation: {conversation}, Item: {item_folder}")
            # print(f"Input:")
            # print(json.dumps(input_content, ensure_ascii=False, indent=4))
            # print(f"Output:")
            # print(json.dumps(output_content, ensure_ascii=False, indent=4))
            # print("=" * 80)
        except json.JSONDecodeError:
            print(f"[Error] Failed to parse JSON for Conversation: {conversation}, Item: {item_folder}")
            error_items.append((conversation, item_folder))
        except Exception as e:
            print(f"[Error] An unexpected error occurred for Conversation: {conversation}, Item: {item_folder}: {e}")
            error_items.append((conversation, item_folder))

# 最后，打印所有没有正确获得输出结果的 items，方便重新定位和请求
if error_items:
    print("\nItems without correct output results:")
    for conversation, item_folder in error_items:
        print(f"- Conversation: {conversation}, Item: {item_folder}")
else:
    print("\nAll items have correct output results.")