# -*- coding: utf-8 -*-

import os
from openai import OpenAI
import json
import time


class LLMs_agent():
    """
    大语言模型请求和结果管理类
    该类用于读取prompt，整合prompt，向大模型发送请求，并将大模型的结果进行JSON解析，返回准确的JSON结果。
    """

    def __init__(self, AKK, base_url, model_type, max_retries=3, open_remote=False):
        """
        初始化函数

        参数：
        AKK (str): API密钥
        base_url (str): API的基础URL
        model_type (str): 使用的模型类型
        max_retries (int): 最大重试次数
        """
        self.open_remote = open_remote
        self.url_base = base_url
        self.client = OpenAI(
            api_key=AKK,
            base_url=base_url,
        )
        self.model_type = model_type
        self.max_retries = max_retries


    def read_prompt(self, path):
        """
        读取prompt文件内容

        参数：
        path (str): prompt文件的路径

        返回：
        str: prompt的内容
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"读取文件 {path} 时出错: {e}")

    def get_safe_prompt(self, prompt):
        """
        获取安全的prompt，添加约束条件，确保大模型输出符合预期格式

        参数：
        prompt (str): 原始的prompt内容

        返回：
        str: 添加了约束的prompt
        """
        safe_prompt = prompt + """
        
- Ensure that your output strictly follows the JSON format.  No need ```json ```.
- Do not include any extra text or polite phrases.
- If the answer is complete, please add [--OK--] at the end. For example:

{
  "audio_samples": [
    {
      "file_name": "audio1.wav",
      "label": "male"
    }
  ]
}
[--OK--]

"""

        return safe_prompt

    def request_llm(self, prompt_text, info="", temperature=0.3, save_path=None):
        """
        请求大模型生成回答，并处理结果

        参数：
        prompt_text (str): 提交给大模型的prompt文本
        info (str): 用于标识的附加信息，通常是文件路径
        temperature (float): 控制生成的随机性
        save_path (str): 如果指定，则保存请求的prompt和大模型的输出

        返回：
        dict: 解析后的JSON结果
        """
        # 如果save_path不为空，则保存请求的prompt至 save_path/input.txt
        if save_path:
            input_path = os.path.join(save_path, 'input.txt')
            with open(input_path, 'w', encoding='utf-8') as f:
                f.write(self.get_safe_prompt(prompt_text))

        # 打印输入大模型的 prompt 文本
        print(self.get_safe_prompt(prompt_text))
        print()

        retries = 0
        while retries < self.max_retries:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_type,
                    messages=[{'role': 'user', 'content': self.get_safe_prompt(prompt_text)}],
                    temperature=temperature,
                )
                llm_answer = completion.choices[0].message.content

                # 检查是否包含[--OK--]标志，如果没有则继续请求，直到达到最大重试次数
                total_answer = llm_answer
                while '[--OK--]' not in total_answer and retries < self.max_retries:
                    # 如果save_path不为空，则保存当前大模型的部分输出至 save_path/output_partial.txt
                    if save_path:
                        output_partial_path = os.path.join(save_path, 'output_partial.txt')
                        with open(output_partial_path, 'w', encoding='utf-8') as f:
                            f.write(total_answer)
                    # 发送"continue"请求大模型继续输出
                    completion = self.client.chat.completions.create(model=self.model_type,messages=[{'role': 'user', 'content': 'continue'}],temperature=temperature,)
                    llm_answer = completion.choices[0].message.content

                    total_answer += llm_answer
                    retries += 1

                # 当结果完成时，删除[--OK--]标志，同时如果有 ```json ``` 包裹内容，则去除
                llm_answer = total_answer.replace('[--OK--]', '').strip()
                if llm_answer.startswith("```json"):
                    llm_answer = llm_answer[7:].strip()
                if llm_answer.endswith("```"):
                    llm_answer = llm_answer[:-3].strip()

                # 如果save_path不为空，则保存大模型结果至 save_path/output.txt
                if save_path:
                    output_path = os.path.join(save_path, 'output.txt')
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(llm_answer)

                # 尝试将回答解析为 JSON
                llm_answer_dict = json.loads(llm_answer)

                return llm_answer_dict  # 返回解析后的结果

            except json.JSONDecodeError as e:
                # JSON解析失败，可能是因为回答不完整或格式不正确
                retries += 1
                print(f"解析JSON时出错，文件 {info}: {e}。正在重试 ({retries}/{self.max_retries})...")
                time.sleep(1)  # 等待一秒再重试

            except Exception as e:
                # 其他异常错误
                retries += 1  # 增加重试次数
                print(f"处理 {info} 时发生错误: {e}。正在重试 ({retries}/{self.max_retries})...")
                time.sleep(10 * retries)

        # 超过最大重试次数，视为失败，返回空字符串
        print(f"在重试 {self.max_retries} 次后，处理 {info} 失败。跳过。")

        # 如果save_path不为空，则保存大模型结果至 save_path/output.txt
        if save_path:
            output_path = os.path.join(save_path, 'output.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(llm_answer if 'llm_answer' in locals() else '')

        return None


if __name__ == '__main__':
    # 从指定模块导入API密钥
    from AK import ant_AK
    base_url = "这里填入请求的url"
    agent = LLMs_agent(AKK, base_url=base_url, model_type="gpt-3.5-turbo", max_retries=3)
    # 构造prompt的路径
    prompt_path = os.path.join('prompt', 'example_prompt.txt')
    # 读取prompt内容
    prompt = agent.read_prompt(prompt_path)
    # 定义对话内容
    conversation = """
role: mainagent
sentence: somebody does not like this kind of process at all.
role: mainagent
sentence: I also did not like it when I was super young but this is fine.
role: mainagent
sentence: But anyway, this is a way to make new friends.
role: interloctr
sentence: Yeah.
role: mainagent
sentence: Do not does not have to be that serious.
role: interloctr
sentence: Yeah.
role: interloctr
sentence: Yeah.
role: interloctr
sentence: Yeah.
role: mainagent
sentence: So it says it is so fine.
    """
    # 请求大模型获取回答
    answer = agent.request_llm(prompt_text=prompt + conversation)
    # 打印大模型的回答
    print(answer)