import time
import os
import pickle
import json
from openai import OpenAI
from prompts import sentiment_prompt, semantic_fine, yuyong_prompt
from AK import AKK
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy  # 导入 copy 模块用于深度复制

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=AKK,  # 替换为你的 API Key
    base_url="https://api.moonshot.cn/v1",
)

k = 10
MAX_Pool = 50
# 数据和输出路径设置
base_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed', 'trn', 'npy_merge_text_k', f'npy_merge_text_k{k}')
output_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed', 'trn', 'LLM_k', f'LLM_k{k}', 'Sentiment_Agents')

# 如果输出路径不存在，则创建路径
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

# 获取文件夹中的对话文件，并取前60个文件
conversation_files = sorted(os.listdir(base_path))[:60]

# 收集所有任务
tasks = []
for conversation in conversation_files:
    # 读取数据
    path = os.path.join(base_path, conversation)
    with open(path, 'rb') as f:
        dialog = pickle.load(f)

    # 为每个对话中的每一项生成任务
    for item_index, item in enumerate(dialog):
        tasks.append((conversation, item_index, item))

# 定义处理单个 item 的函数
def process_item(task):
    # 解压任务
    conversation, item_index, item = task
    conversation = conversation.replace('.pkl', '')  # 去掉 .pkl 后缀
    # 构建输出路径
    conversation_folder = os.path.join(output_path, conversation)
    if not os.path.exists(conversation_folder):
        os.makedirs(conversation_folder)
    item_folder_name = f'item{item_index + 1:03d}'  # 命名为 item001, item002...
    item_folder = os.path.join(conversation_folder, item_folder_name)
    if not os.path.exists(item_folder):
        os.makedirs(item_folder)

    # 定义输入输出文件路径
    input_file = os.path.join(item_folder, 'input.txt')
    output_file = os.path.join(item_folder, 'output.txt')

    # 检查是否已经存在输出文件，避免重复处理
    if os.path.exists(output_file):
        print(f"Output for {conversation}/{item_folder_name} already exists. Skipping.")
        return

    # 构建请求内容
    content = "\n\n ```plain text\n\n"
    for one in item:
        """
        one:
            - 'role' 表示说话者角色
            - 'sentence' 表示句子内容
            - 'wordtimestamp' 表示时间戳
        """
        content += "           || \n\n ||            role: "
        content += one['role'] + "\n "
        content += "sentence: "
        content += one['sentence'] + "\n "
        content += "          || \n\n ||     "
    content += f"```      \n\n      !!!! Note: There are a total of {k} sentences, and your response must consist of {k} sentences. !!! "

    # 构建消息体
    mess = [
        {
            "role": "user",
            "content": sentiment_prompt + content
        },
    ]

    # 保存请求内容到 input.txt，以 JSON 格式
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(mess, f, ensure_ascii=False, indent=4)

    # 初始化变量
    max_retries = 3  # 最大重试次数
    retries = 0  # 当前重试次数
    success = False  # 用于标记是否成功完成
    max_continue = 3  # 最大 continue 次数

    while retries < max_retries:
        try:
            continue_attempts = 0  # 当前 continue 次数
            current_mess = copy.deepcopy(mess)  # 深度复制消息列表，避免修改原始消息
            full_llm_answer = ''  # 重置完整回答字符串

            while continue_attempts <= max_continue:
                # 请求模型生成回答
                completion = client.chat.completions.create(
                    model="moonshot-v1-auto",
                    messages=current_mess,
                    temperature=0.3,
                )
                llm_answer = completion.choices[0].message.content
                # 将当前回答追加到完整回答中
                full_llm_answer += llm_answer

                # 尝试将完整回答解析为 JSON
                try:
                    llm_answer_dict = json.loads(full_llm_answer)

                    # 打印解析的文本，验证是否成功
                    print(f"Parsed text for {conversation}/{item_folder_name}: {llm_answer_dict[0]['Text']}")

                    # 检查解析出来的回答数量是否与原始句子数相同
                    if len(llm_answer_dict) != len(item):
                        print(f"Mismatch in sentence count for {conversation}/{item_folder_name}. Retrying.")
                        print(f"Original sentences: {len(item)}, LLM sentences: {len(llm_answer_dict)}")
                        break
                        continue_attempts += 1
                        # 超过最大尝试次数，跳出循环
                        if continue_attempts > max_continue:
                            print(f"Exceeded max continue attempts for {conversation}/{item_folder_name}.")
                            break
                        else:
                            # 更新消息体，加入 assistant 上次的回答和 user 的 'continue' 提示
                            current_mess.append({
                                "role": "assistant",
                                "content": llm_answer
                            })
                            current_mess.append({
                                "role": "user",
                                "content": f"Mismatch in sentence count. There are {k} sentences in total, but your answer doesn't match. Please provide the correct result directly without any apologies or polite phrases."
                            })
                            continue  # 继续请求
                    else:
                        # 成功解析且句子数匹配，保存输出内容到 output.txt，以 JSON 格式
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(llm_answer_dict, f, ensure_ascii=False, indent=4)
                        success = True  # 标记成功
                        break  # 解析成功，退出 continue 循环

                except json.JSONDecodeError as e:
                    # 解析失败，可能是因为回答不完整，需要继续请求
                    continue_attempts += 1
                    if continue_attempts > max_continue:
                        print(f"Exceeded max continue attempts for {conversation}/{item_folder_name}.")
                        break  # 超过最大 continue 次数，退出 continue 循环
                    else:
                        # 更新消息体，加入 assistant 上次的回答和 user 的 'continue' 提示
                        current_mess.append({
                            "role": "assistant",
                            "content": llm_answer
                        })
                        current_mess.append({
                            "role": "user",
                            "content": "continue"
                        })
                        continue  # 继续请求

            if success:
                break  # 如果成功，退出 retries 循环
            else:
                retries += 1  # 未成功，增加重试次数
                time.sleep(1)  # 等待一秒再重试

        except Exception as e:
            retries += 1  # 增加重试次数
            print(f"Error processing {conversation}/{item_folder_name}: {e}. Retrying ({retries}/{max_retries})...")
            time.sleep(1)  # 等待一秒再重试
    else:
        # 超过最大重试次数，视为失败
        print(f"Failed to process {conversation}/{item_folder_name} after {max_retries} retries. Skipping.")

# 使用线程池并行处理任务，并显示进度条
start_time = time.time()
with ThreadPoolExecutor(max_workers=MAX_Pool) as executor:
    futures = [executor.submit(process_item, task) for task in tasks]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing tasks"):
        try:
            future.result()
        except Exception as e:
            print(f"Task failed with exception: {e}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"All tasks completed in {elapsed_time:.2f} seconds.")
