import pickle
import os
import math
import time

from llm_tools import LLMs_agent
from AK import AKK
import numpy as np
agent = LLMs_agent(AKK, base_url='', model_type='o1-preview-0912')
"""
    prompt_path = os.path.join('..', 'Method_01_LLMs_Human', 'prompt', 'example_prompt.txt')
    prompt = agent.read_prompt(prompt_path)
    answer = agent.request_llm(prompt_text=prompt)
    返回json格式
"""


class Human_History():
    def __init__(self, conversation):
        """
        初始化Human_History类

        参数：
            conversation: 对话的内容，是一个字典列表，每个元素包含：
                 - role: 说话者的身份 ('mainagent' 或 'interloctr')
                 - sentence: 说话者的句子
                 - wordtimestamp: 句子中每个单词的时间戳信息
        """
        self.conversation = conversation
        self.conversation_length = len(conversation)

    def get_n_sentences(self, n):
        """
        获取对话中的第n个句子

        参数：
            n: 需要获取的句子索引（从0开始）

        返回：
            对应的句子字典，包含'sentence', 'role', 'wordtimestamp'等信息
        """
        if n >= self.conversation_length:
            n = self.conversation_length - 1
        if n < 0:
            n = 0
        return self.conversation[n]

    def get_n_sentences_history(self, n, role, including_2_part, mode):
        """
        获取对话历史记录

        参数：
            n: 当前对话的轮次索引
            role: 当前角色 ('mainagent' 或 'interloctr')
            including_2_part: 是否包含对方的对话内容
            mode: 返回历史对话的模式
                - 'All': 返回第n轮对话及之前的所有对话
                - 'KN':  返回第n轮及前N-1轮对话 (例如: "k10" 返回前9轮)
                - 'Ebbinghaus': 根据艾宾浩斯遗忘曲线，返回加权后的历史对话

        返回：
            包含当前对话第n轮的句子以及相应的历史对话信息的字典：
            {
                'focus': 当前第n个句子,
                'history': 包含历史对话信息的列表
            }
        """

        # 根据including_2_part决定是否过滤对方的对话
        if including_2_part:
            history_conversation = self.conversation[:n]
        else:
            history_conversation = [conv for conv in self.conversation[:n] if conv['role'] == role]

        # 获取第n轮的句子
        if n >= len(history_conversation):
            n = len(history_conversation) - 1
        if n < 0:
            n = 0
        current_sentence = history_conversation[n]

        # 根据不同的模式处理历史对话
        if mode.startswith('k') and mode[1:].isdigit():
            # KN 模式: 提取前N-1轮对话
            K = int(mode[1:])
            history_conversation = history_conversation[max(0, n - (K - 1)):n + 1]
        elif mode == 'Ebbinghaus':
            # Ebbinghaus 模式: 根据艾宾浩斯遗忘曲线进行加权筛选
            history_conversation = self.apply_ebbinghaus(history_conversation)
        elif mode != 'All':
            raise ValueError(f"未知的模式: {mode}")

        return {
            'focus': current_sentence,
            'history': history_conversation
        }, len(history_conversation) - 1

    def apply_ebbinghaus(self, history_conversation):
        """
        根据艾宾浩斯遗忘曲线来权重筛选历史对话，越早的对话被遗忘概率越高

        参数：
            history_conversation: 需要处理的历史对话列表

        返回：
            根据遗忘曲线加权筛选后的历史对话列表
        """

        # 定义遗忘曲线函数：时间越久，概率越高
        def forgetting_curve(index, total):
            """
            遗忘曲线函数，返回一个介于0到1之间的值，用于加权判断是否保留对话。
            index: 当前对话的索引（越靠近0表示越早的对话）
            total: 历史对话的总数（越靠近total表示越近期的对话）
            """
            # 使用一个简单的指数衰减函数来模拟遗忘曲线
            return math.exp(-(total - index) / total)

        # 筛选历史对话，使用遗忘曲线函数进行加权删除
        total_conversations = len(history_conversation)
        filtered_conversation = [
            history_conversation[i] for i in range(total_conversations)
            if forgetting_curve(i, total_conversations) > 0.2  # 设置一个阈值，0.2以下的内容容易遗忘
        ]
        return filtered_conversation


class ChappieLLMs:
    def __init__(self, name, role, conversation_path, cache_path=None, history_mode='All', save_path=None):
        self.name = name
        # 对话文件路径
        self.conversation_path = conversation_path
        # 加载对话
        with open(self.conversation_path, 'rb') as f:
            self.conversation = pickle.load(f)
        # 管理对话记忆
        self.history = Human_History(self.conversation)
        self.history_mode = history_mode
        # 管理对话帧长
        self.conversation_length = self.conversation[-1]['wordtimestamp'][-1][1]
        # 状态文件路径
        self.state = {}
        if cache_path:
            self.state_file = os.path.join(cache_path, f'{role}_state.pkl')
        self.save_path = save_path

        # prompt 管理
        self.summary_prompt = agent.read_prompt(os.path.join('prompt', 'summary_prompt.txt'))
        self.global_intention_prompt = agent.read_prompt(os.path.join('prompt', 'global_intention_prompt.txt'))
        self.emotion_prompt = agent.read_prompt(os.path.join('prompt', 'emotion_prompt.txt'))
        self.local_intention_prompt = agent.read_prompt(os.path.join('prompt', 'local_intention_prompt.txt'))
        # self.trigger_prompt = agent.read_prompt(os.path.join('..', 'Method_01_LLMs_Human', 'prompt', 'trigger_prompt.txt'))
        self.motion_trigger_prompt = agent.read_prompt(os.path.join('prompt', 'motiontrigger_prompt.txt'))
        self.topic_prompt = agent.read_prompt(os.path.join('prompt', 'topic_prompt.txt'))

    def read_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'rb') as f:
                self.state = pickle.load(f)

    def get_max_n(self, role, including_2_part):
        if including_2_part:
            history_conversation = self.conversation
        else:
            history_conversation = [conv for conv in self.conversation if conv['role'] == role]
        return len(history_conversation) - 1

    def get_info(self, cate, prompt, n, role='mainagent', including_2_part=True):
        """
        获取第n轮对话的信息

        参数：
            n: 对话轮数
            role: 当前角色 ('mainagent' 或 'interloctr')
            including_2_part: 是否包含对方的对话内容
        """
        # 更新state，确保目前使用的state是最新的
        self.read_state()

        # 获取对话历史
        conversation, max_n = self.history.get_n_sentences_history(n, role, including_2_part, self.history_mode)

        if n > max_n:
            n = max_n
        if n < 0:
            n = 0

        # 初始化模式下的缓存
        if 'including_2_part' not in self.state:
            self.state['including_2_part'] = {}

        # 初始化具体模式的缓存
        if self.history_mode not in self.state['including_2_part']:
            self.state['including_2_part'][self.history_mode] = {}

        # 初始化第n轮的缓存
        if n not in self.state['including_2_part'][self.history_mode]:
            self.state['including_2_part'][self.history_mode][n] = {}

        # 当前轮对话的起始/结束
        start_frame = conversation['focus']['wordtimestamp'][0][0]
        end_frame = conversation['focus']['wordtimestamp'][-1][1]

        if cate == 'Global_Intention':
            # 打印对话
            print(" \n".join(
                ['role:' + i['role'] + '\n' + 'sentence:' + i['sentence'] + '\n' for i in conversation['history']]))

        # 检查缓存，直接返回缓存里的人格特征
        if cate in self.state['including_2_part'][self.history_mode][n] and self.state['including_2_part'][self.history_mode][n][cate] != '':
            print(f"从缓存中获取{cate}")
            n_info = self.state['including_2_part'][self.history_mode][n][cate]
        else:
            # 请求大模型生成人格特征
            # time.sleep(10)
            n_info = agent.request_llm(prompt + " \n".join(['role:' + i['role'] + '\n' + 'sentence:' + i['sentence'] + '\n' for i in conversation['history']]) + ' \n ', save_path=self.save_path)
            if n_info:
                # 将生成的人格特征保存到缓存
                self.state['including_2_part'][self.history_mode][n][cate] = n_info
                # 同时保存到磁盘，确保持久化
                os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
                with open(self.state_file, 'wb') as f:
                    pickle.dump(self.state, f)
                print(f"大模型生成{cate}并保存到缓存")
            else:
                return None

        # 输出人格特征
        print(n_info[cate])
        return {
            'start_frame': start_frame,
            'end_frame': end_frame,
            cate: n_info[cate]
        }

    def get_human_info(self, n, role='mainagent', including_2_part=True):
        global_intention = self.get_info('Global_Intention', self.global_intention_prompt, n, role, including_2_part)
        summary = self.get_info('Summary', self.summary_prompt, n, role, including_2_part)
        emotion = self.get_info('Emotion', self.emotion_prompt, n, role, including_2_part)
        LocalIntention = self.get_info('LocalIntention', self.local_intention_prompt, n, role, including_2_part)
        # trigger_prompt = self.get_info('Trigger', self.trigger_prompt, n, role, including_2_part)
        motion_trigger_prompt = self.get_info('SentenceMotionTrigger', self.motion_trigger_prompt, n, role, including_2_part)
        topic_prompt = self.get_info('TopicSentenceLabel', self.topic_prompt, n, role, including_2_part)


        # assert personality['start_frame'] == summary['start_frame']
        # assert personality['end_frame'] == summary['end_frame']

        try:
            return {
                'start_frame': summary['start_frame'],
                'end_frame': summary['end_frame'],
                'Summary': summary['Summary'],
                'Global_Intention': global_intention['Global_Intention'],
                'LocalIntention': LocalIntention['LocalIntention'],
                'Emotion': emotion['Emotion'],
                # 'Trigger': trigger_prompt['Trigger'],
                'SentenceMotionTrigger': motion_trigger_prompt['SentenceMotionTrigger'],
                'TopicSentenceLabel': topic_prompt['TopicSentenceLabel'],
            }
        except Exception as e:
            return None

import io
import string
from tqdm import tqdm

import numpy as np
from multiprocessing import Pool, cpu_count


def process_chunk(chunk):
    """处理文件块中的词向量"""
    data = {}
    for line in chunk:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(v) for v in tokens[1:]])
    return data


def split_file_into_chunks(fname, num_chunks):
    """将文件分为若干块以供并行处理"""
    with io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        # 跳过第一行 (词汇量和维度)
        fin.readline()
        lines = fin.readlines()

    # 将文件分成num_chunks块
    chunk_size = len(lines) // num_chunks
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    return chunks


def load_word_vectors(fname, workers=None):
    """使用多进程并行加载词向量"""
    print("Loading word2vector in parallel...")

    # 如果没有指定workers，默认使用所有可用的CPU核心
    if workers is None:
        workers = cpu_count()

    # 分割文件为多个块
    chunks = split_file_into_chunks(fname, workers)

    # 使用进程池进行并行处理
    with Pool(processes=workers) as pool:
        # tqdm 进度条显示进程池的工作情况
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))

    # 合并所有结果
    word_vectors = {}
    for result in results:
        word_vectors.update(result)

    return word_vectors


def turn_w2v(sentence, word2vector):

    textfeatures = []
    for wi, raw_word in enumerate(sentence):
        word = raw_word.translate(str.maketrans('', '', string.punctuation))
        word = word.strip()
        word = word.replace("  ", " ")

        if len(word) > 0:
            if word[0] == " ":
                word = word[1:]

        if " " in word:
            ww = word.split(" ")
            for j, w in enumerate(ww):
                vector = word2vector.get(w)
                if vector is not None:
                    textfeatures.append(vector)
        else:
            pass
            vector = word2vector.get(word)
            if vector is not None:
                textfeatures.append(vector)

    # 全部加和，保证输出[300]
    textfeatures = np.array(textfeatures).sum(axis=0)
    return textfeatures


if __name__ == '__main__':
    # k = 10
    # conversation_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed',
    #                                  'trn', 'npy_merge_text', 'trn_2023_v0_000.pkl')
    # cache_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed',
    #                           'trn', 'Method_01', 'Person_State', os.path.basename(conversation_path).replace('.pkl', ''))
    # human = ChappieLLMs(name='main-agent', role='mainagent', conversation_path=conversation_path, cache_path=cache_path)
    # print(human.get_human_info(n=10))

    # ../../resources\checkpoints/crawl-300d-2M.vec
    # w2v = load_word_vectors(os.path.join('..', '..', 'resources', 'checkpoints', 'crawl-300d-2M.vec'), workers=10)

    mode = 'val'

    w2v = {}

    # limit_training_data = 20
    from glob import glob
    limit_training_data = len(glob(os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed',
                                         mode, 'npy_merge_text', f'*.pkl')))
    human_llm_info = []
    human_llm_info_sum = []
    human_llm_info_intention = []
    human_llm_info_emotion = []
    human_llm_info_local_intention = []
    human_llm_info_trigger = []
    human_llm_info_motion_trigger = []
    human_llm_info_motion_topic = []

    error = []
    for conversation_idx in tqdm(range(limit_training_data)):
        conversation_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed',
                                         mode, 'npy_merge_text', f'{mode}_2023_v0_{conversation_idx:03d}.pkl')
        cache_path = os.path.join('..', '..', 'resources', 'datasets', 'twh_genea2023_dataset', 'processed',
                                  mode, 'Method_01', 'Person_State',
                                  os.path.basename(conversation_path).replace('.pkl', ''))
        human = ChappieLLMs(name='main-agent', role='mainagent', conversation_path=conversation_path,
                                     cache_path=cache_path)
        max_n = human.get_max_n(role='mainagent', including_2_part=True)

        info = human.get_human_info(max_n)
        if info == None:
            error.append(conversation_idx)
            continue
        summary = info['Summary']
        global_intention = info['Global_Intention']
        LocalIntention = info['LocalIntention']
        emotion = info['Emotion']
        # trigger = info['Trigger']
        motion_trigger = info['SentenceMotionTrigger']
        topic = info['TopicSentenceLabel']

        human_llm_info_sum.append(summary)
        human_llm_info_intention.append(global_intention)
        human_llm_info_emotion.append(emotion)
        human_llm_info_local_intention.append(LocalIntention)
        # human_llm_info_trigger.append(trigger)
        human_llm_info_motion_trigger.append(motion_trigger)
        human_llm_info_motion_topic.append(topic)

        summary_vec = turn_w2v(summary.split(" "), w2v)
        global_intention_vec = turn_w2v(global_intention.split(" "), w2v)

        # (300,)
        human_llm_info.append(summary_vec)

    human_llm_info = np.array(human_llm_info)

    print(f'失败: {error}')
    # 保存pickle到当前目录下 ./human_llm_info_sum 和 ./human_llm_info_intention

    import pickle
    # 保存到 ./human_llm_info_sum.pkl
    with open(f'../Ours/human_llm_info_sum_{mode}.pkl', 'wb') as f:
        pickle.dump(human_llm_info_sum, f)

    with open(f'../Ours/human_llm_info_intention_{mode}.pkl', 'wb') as f:
        pickle.dump(human_llm_info_intention, f)

    with open(f'../Ours/human_llm_info_emotion_{mode}.pkl', 'wb') as f:
        pickle.dump(human_llm_info_emotion, f)

    with open(f'../Ours/human_llm_info_local_intention_{mode}.pkl', 'wb') as f:
        pickle.dump(human_llm_info_local_intention, f)

    # with open(f'../Ours/human_llm_info_tigger_{mode}.pkl', 'wb') as f:
    #     pickle.dump(human_llm_info_trigger, f)

    with open(f'../Ours/human_llm_info_motion_tigger_{mode}.pkl', 'wb') as f:
        pickle.dump(human_llm_info_motion_trigger, f)

    with open(f'../Ours/human_llm_info_topicSentence_{mode}.pkl', 'wb') as f:
        pickle.dump(human_llm_info_motion_topic, f)

    # print(human_llm_info_sum)