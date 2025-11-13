import time
from openai import OpenAI
import pickle
import os
import time
import json
from prompts.prompt import semantic_analysis, Pragmatic_Analysis_Prompt_V2, sentment_prompt
from AK import kimi_AK

client = OpenAI(
    api_key=kimi_AK,  # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
    base_url="https://api.moonshot.cn/v1",
)

k = 10
base_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed', 'val', 'npy_merge_text_k', f'npy_merge_text_k{k}')
conversation_files = sorted(os.listdir(base_path))

count = 0
for i, conversation in enumerate(conversation_files[3:]):
    # 读数据
    path = os.path.join(base_path, conversation)
    with open(path, 'rb') as f:
        dialog = pickle.load(f)

    for item in dialog:
        content = "*" * 10 + "\n "
        for one in item:
            content += "role: "
            content += one['role'] + "\n "
            content += "sentence: "
            content += one['sentence'] + "\n "
            content += "*" * 10 + "\n "

        # 拿到了一个数据项 k轮对话
        # print(content)
        mess = [
                {"role": "user",
                 "content": sentment_prompt + content
                 },
            ]
        # print(mess)

        output_llms = ""
        output_llms += content
        output_llms += json.dumps(mess, ensure_ascii=False)
        output_llms += '\n\n================= 结果 ================\n\n'

        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=mess,
            temperature=0.3,
        )

        llm_answer = completion.choices[0].message.content
        output_llms += llm_answer

        print(output_llms)

        llm_answer_dict = json.loads(llm_answer)

        # 保存结果： output_llms


        break

    break
