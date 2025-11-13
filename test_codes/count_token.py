"""
    用来计算token数

"""


import requests
import json
import os
import pickle
from tqdm import tqdm
from AK import kimi_AK

# 从环境变量中获取 API 密钥
api_key = kimi_AK  # 确保在运行脚本前设置该环境变量

# 定义请求 URL
url = "https://api.moonshot.cn/v1/tokenizers/estimate-token-count"

# 定义请求头，包括 API 密钥
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"  # 将 API 密钥添加到请求头
}

k = 10
base_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed', 'val', 'npy_merge_text_k', f'npy_merge_text_k{k}')
conversation_files = sorted(os.listdir(base_path))
# print(conversation_files)

count = 0
for i, conversation in enumerate(tqdm(conversation_files)):
    # 读数据
    path = os.path.join(base_path, conversation)
    with open(path, 'rb') as f:
        dialog = pickle.load(f)

    for item in dialog:

        content = "*" * 50 + "\n"
        for one in item:
            content += "role: "
            content += one['role'] + "\n "
            content += "sentence: "
            content += one['sentence'] + "\n "
            content += "*" * 50 + "\n "

        # 拿到了一个数据项 k轮对话
        # print(content)
        mess = [
                {"role": "user",
                 "content": """
                Provide valid JSON output.

                You are a nonverbal communication expert who can keenly detect emotions and intentions in multi-person conversations and accurately infer what actions each person might be doing while speaking.

                You will be given a conversation between two people.

                Your task is to produce multiple action sequences, displaying them continuously as (text, intention, emotion, action).

                Let’s think step by step. For each line of dialogue, analyze and provide:

                Text: The words spoken by the person. The speech script.
                KeyWords: Identify words that might be keywords.
                Thinking: Think about the steps, what emotions and intentions the speaker has, and what motions the speaker does because of what words. For example, if you say "you broke my water bottle and I'm angry", then there might be the keyword "angry", and the corresponding emotion is "anger", and the corresponding action is the fist of the hand clenched in the air and extended, the head has a big movement of excitement, and the body trembles.
                Intention: The underlying intent behind the dialogue.
                Emotion: The emotion conveyed by this dialogue. Please use one of the following 48 categories: high arousal, Excited, Delighted, Tense, Angry, low arousal, Tired, Bored, Relaxed, Calm, positive valence, Happy, Content, Relaxed, Calm, negative valence, Angry, Frustrated, Depressed, Tired, neutral, Joy, Serenity, Ecstasy, Trust, Acceptance, Admiration, Fear, Apprehension, Terror, Surprise, Distraction, Amazement, Sadness, Pensiveness, Grief, Disgust, Boredom, Loathing, Anger, Annoyance, Rage, Anticipation, Interest, Vigilance.
                Motion: Possible corresponding motions in the speech script.  Note that the speaker is standing face to face with the listener.

                Output your analysis in valid JSON format with the following structure:
                [{
                    "Text": "string",
                    "KeyWords": ["string"],
                    "Thinking": "string",
                    "Intention": "string",
                    "Emotion": "string (from the provided categories)",
                    "Motion": "string",
                },
                ...
                ]
                Ensure that your output strictly follows the JSON format. You don't need to show ```json.

                Please answer the results directly, without any polite language.

                Please analyze the following conversations: \n
                """ + content
                 },
            ]

        # 定义请求体
        data = {
            "model": "moonshot-v1-8k",
            "messages": mess
        }

        # 发送 POST 请求
        response = requests.post(url, headers=headers, data=json.dumps(data))

        # 打印响应结果
        if response.status_code == 200:
            print("Token count estimation:", response.json()['data']['total_tokens'])
        else:
            print(f"Failed to get token count. Status code: {response.status_code}, Response: {response.text}")

        count += response.json()['data']['total_tokens']

print(count)