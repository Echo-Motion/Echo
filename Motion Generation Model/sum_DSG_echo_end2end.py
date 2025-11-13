import argparse
import torch
import yaml
from pprint import pprint
import h5py
import numpy as np
from easydict import EasyDict
import os
from torch.utils.data import DataLoader
from sum_DSG_mdm import MDM
from model_util import create_gaussian_diffusion, load_model_wo_clip
from diffusion import logger
from diffusion.fp16_util import MixedPrecisionTrainer
from torch.optim import AdamW
from diffusion.resample import create_named_schedule_sampler
from diffusion.resample import LossAwareSampler
from tqdm import tqdm
import functools
import blobfile as bf
import time
import string
from winniyCode.push_WeCom import text_message

OpenIntend = False
OpenSenti = False
resume_step = 0
re_ckp_path = os.path.join('', '..', 'baseline', 'baselineDSG', 'model000050000.pt')
re_ckp_opt_path = None

mode = 'trn'
if mode == 'val':
    worker = 3
    nroot = "feature_new"
else:
    worker = 20
    nroot = os.path.join('', '..', 'baseline', 'feature_new')

"""
    args
"""
def parse_args():
    parser = argparse.ArgumentParser(description='DiffuseStyleGesture')
    parser.add_argument('--config', default='config/sum_DSG.yml')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--no_cuda', type=list, default=['0'])
    parser.add_argument('--intend', action='store_true')
    parser.add_argument('--senti', action='store_true')

    args = parser.parse_args()
    return args

"""
    新加部分
"""
import io
from Method_01_LLMs_Human.ChappieLLM import ChappieLLMs

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

    if mode == 'val':
        # 随机返回维度为300的向量
        return np.random.randn(300)

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

"""
    dataset
"""

class SpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, main_h5file_1, main_h5file_2=None, inter_h5file_1=None, inter_h5file_2=None, sequence_length=30 * 5, npy_root="../../process"):

        gesture_mean = np.load(os.path.join(npy_root, "gesture_mean.npy"))
        gesture_std = np.load(os.path.join(npy_root, "gesture_std.npy"))
        self.sequence_length = sequence_length

        # main
        self.h5 = h5py.File(main_h5file_1, "r")
        self.len = len(self.h5.keys())
        # self.id = [self.h5[str(i)]["speaker_id"][:] for i in range(len(self.h5.keys()))]
        self.audio = [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
        self.text = [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        # if OpenIntend:
        #     self.intend = [self.h5[str(i)]["intend"][:] for i in range(len(self.h5.keys()))]
        # if OpenSenti:
        #     self.senti = [self.h5[str(i)]["senti"][:] for i in range(len(self.h5.keys()))]
        self.gesture = [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        self.motion_dim = 744
        self.h5.close()

        if main_h5file_2 is not None:
            self.h5 = h5py.File(main_h5file_2, "r")
            self.len = len(self.h5.keys())
            # self.id += [self.h5[str(i)]["speaker_id"][:] for i in range(len(self.h5.keys()))]
            self.audio += [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
            self.text += [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
            # if OpenIntend:
            #     self.intend += [self.h5[str(i)]["intend"][:] for i in range(len(self.h5.keys()))]
            # if OpenSenti:
            #     self.senti += [self.h5[str(i)]["senti"][:] for i in range(len(self.h5.keys()))]
            self.gesture += [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
            self.h5.close()

        # self.h5 = h5py.File(inter_h5file_1, "r")
        # self.len = len(self.h5.keys())
        # # self.id += [self.h5[str(i)]["speaker_id"][:] for i in range(len(self.h5.keys()))]
        # self.audio_inter = [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
        # self.text_inter = [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        # if OpenIntend:
        #     self.intend_inter = [self.h5[str(i)]["intend"][:] for i in range(len(self.h5.keys()))]
        # if OpenSenti:
        #     self.senti_inter = [self.h5[str(i)]["senti"][:] for i in range(len(self.h5.keys()))]
        # # self.gesture_inter = [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        # self.h5.close()

        # if inter_h5file_2 is not None:
        #     self.h5 = h5py.File(inter_h5file_2, "r")
        #     self.len = len(self.h5.keys())
        #     # self.id += [self.h5[str(i)]["speaker_id"][:] for i in range(len(self.h5.keys()))]
        #     self.audio_inter += [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
        #     self.text_inter += [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        #     if OpenIntend:
        #         self.intend_inter += [self.h5[str(i)]["intend"][:] for i in range(len(self.h5.keys()))]
        #     if OpenSenti:
        #         self.senti_inter += [self.h5[str(i)]["senti"][:] for i in range(len(self.h5.keys()))]
        #     # self.gesture_inter += [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        #     self.h5.close()

        self.gesture_vel = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in self.gesture]
        self.gesture_acc = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in self.gesture_vel]

        print("Total clips:", len(self.gesture))
        self.segment_length = sequence_length

        ################################### 新加部分
        if mode == 'trn':
            w2v = load_word_vectors(os.path.join('', '..', 'resources', 'checkpoints', 'crawl-300d-2M.vec'), worker)
        else:
            w2v = {}

        import pickle
        with open('human_llm_info_sum.pkl', 'rb') as f:
            human_llm_info_sum = pickle.load(f)

        human_llm_info = []
        for info in tqdm(human_llm_info_sum):

            vec = turn_w2v(info.split(" "), w2v)
            human_llm_info.append(vec)

        # (limit_training_data, 300)
        self.human_llm_info = np.array(human_llm_info)

    def __len__(self):
        return len(self.gesture)

    def __getitem__(self, idx):

        human_info = self.human_llm_info[idx]
        human_info = np.tile(human_info, (self.segment_length, 1))
    
        total_frame_len = self.audio[idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length

        audio = self.audio[idx][start_frame:end_frame]
        text = self.text[idx][start_frame:end_frame]
        # audio_inter = self.audio_inter[idx][start_frame:end_frame]
        # text_inter = self.text_inter[idx][start_frame:end_frame]
        # if OpenIntend:
            # intend = self.intend[idx][start_frame:end_frame]
            # intend_inter = self.intend_inter[idx][start_frame:end_frame]
        # if OpenSenti:
            # senti = self.senti[idx][start_frame:end_frame]
            # senti_inter = self.senti_inter[idx][start_frame:end_frame]

        textaudio = np.concatenate((audio, text), axis=-1)
        # if OpenIntend:
        #     textaudio = np.concatenate((textaudio, intend, intend_inter), axis=-1)
        # if OpenSenti:
        #     textaudio = np.concatenate((textaudio, senti, senti_inter), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        posrat = self.gesture[idx][start_frame:end_frame]
        vel = self.gesture_vel[idx][start_frame:end_frame]
        acc = self.gesture_acc[idx][start_frame:end_frame]
        gesture = np.concatenate((posrat, vel, acc), axis=-1)
        gesture = torch.FloatTensor(gesture)
        speaker = np.zeros([17])
        speaker[0] = 1      # dummy speaker
        speaker = torch.FloatTensor(speaker)

        human_info = torch.FloatTensor(human_info)
        return textaudio, gesture, speaker, human_info  # textaudio torch.Size([150, 3808]) | gesture torch.Size([150, 2232]) | speaker torch.Size([17])

class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)

class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        return iter(range(self.min_id, self.max_id))

"""
    model
"""
def create_model_and_diffusion(args):
    model = MDM(modeltype='', njoints=2232, nfeats=1, cond_mode='cross_local_attention4_style1', action_emb='tensor', audio_feat=args.audio_feat,
                arch='trans_enc', latent_dim=512, n_seed=30, cond_mask_prob=0.1, device=device_name)

    if re_ckp_path is not None:
        state_dict = torch.load(re_ckp_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

    diffusion = create_gaussian_diffusion()
    return model, diffusion

"""
    Train
"""
class TrainLoop:
    def __init__(self, args, model, diffusion, device, data=None):
        self.args = args
        self.data = data
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode    #: cross_local_attention4_style1
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.log_interval = args.log_interval #: 50
        self.weight_decay = args.weight_decay
        self.save_step = args.save_step


        self.lr_anneal_steps = args.lr_anneal_steps
        self.microbatch = args.batch_size  # deprecating this option
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option

        self.step = 0
        self.resume_step = resume_step
        self.global_batch = self.batch_size
        self.num_epochs = 50000000
        self.n_seed = 30

        self.sync_cuda = torch.cuda.is_available()
        self.mp_trainer = MixedPrecisionTrainer(        # 混精度加速训练
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.save_dir = args.save_dir

        self.device = device
        self.opt = AdamW([
            {'params': self.mp_trainer.master_params, 'lr':self.lr, 'weight_decay':self.weight_decay}
        ])
        if re_ckp_opt_path is not None:
            # 加载优化器状态,允许部分参数不匹配
            self.opt.load_state_dict(torch.load(re_ckp_opt_path, map_location='cpu'))

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False
        self.ddp_model = self.model
        self.mask_train = (torch.zeros([self.batch_size, 1, 1, args.n_poses]) < 1).to(self.device)
        self.mask_test = (torch.zeros([1, 1, 1, args.n_poses]) < 1).to(self.device)
        self.mask_local_train = torch.ones(self.batch_size, args.n_poses).bool().to(self.device)
        self.mask_local_test = torch.ones(1, args.n_poses).bool().to(self.device)

        self.start_time = time.time()

    def run_loop(self):

        for batch in tqdm(self.data):

            # 检查训练时间是否超过15分钟
            if time.time() - self.start_time > 15 * 60:  # 15分钟 = 15 * 60秒
                text_message("训练完成")
                break

            cond_ = {'y':{}}
            # pose_seq.shape torch.Size([2, 150, 2232])
            wavlm, pose_seq, style, human_info = batch
            motion = pose_seq.permute(0, 2, 1).unsqueeze(2).to(self.device, non_blocking=True)
            # motion.shape torch.Size([2, 2232, 1, 150])

            cond_['y']['seed'] = motion[..., 0:self.n_seed]
            cond_['y']['style'] = style.to(self.device, non_blocking=True)
            cond_['y']['human_info'] = human_info.to(self.device, non_blocking=True)
            cond_['y']['mask_local'] = self.mask_local_train
            cond_['y']['gesture'] = pose_seq.to(self.device, non_blocking=True)     # attention2 (Tactron)
            cond_['y']['audio'] = wavlm.to(torch.float32)[:, self.n_seed:].to(self.device, non_blocking=True)       # attention4/5
            cond_['y']['mask'] = self.mask_train        # [..., self.n_seed:]

            self.run_step(motion, cond_)
            if self.step % self.log_interval == 0:
                for k,v in logger.get_current().name2val.items():
                    if k == 'loss':
                        print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, v))

            if self.step % self.save_step == 0:
                self.save()

            self.step += 1

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)      # torch.Size([64, 251, 1, 196]) cond['y'].keys() dict_keys(['mask', 'lengths', 'text', 'tokens'])
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, ch, image_size, image_size]      # x_start, (2, 135, 1, 240)
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond,
                dataset='kit'
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)

            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def main(args):
    print("Loading dataset into memory ...")
    trn_dataset = SpeechGestureDataset(main_h5file_1=os.path.join(nroot, f'{mode}_main-agent_bert_v0.h5'),
                                       # main_h5file_2=os.path.join(args.root_dir, 'val_main-agent_bert_v0.h5'),
                                    #    inter_h5file_1=os.path.join(nroot, f'{mode}_interloctr_bert_v0.h5'),
                                       # inter_h5file_2=os.path.join(args.root_dir, 'val_interloctr_bert_v0.h5'),
                                       # os.path.join(args.root_dir, 'trn_main-agent_v0.h5'),
                                       npy_root=nroot)

    train_loader = DataLoader(trn_dataset,
                              sampler=RandomSampler(0, len(trn_dataset)),
                              batch_size=args.batch_size,
                              pin_memory=True,
                              drop_last=False)

    pprint('*'*80)
    pprint(args.model_save_path)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    pprint('create model')
    model, diffusion = create_model_and_diffusion(args)
    model.to(mydevice)
    pprint(model)

    TrainLoop(args, model, diffusion, mydevice, data=train_loader).run_loop()


if __name__ == '__main__':
    args = parse_args()
    device_name = 'cuda:' + args.gpu
    mydevice = torch.device(device_name)
    torch.cuda.set_device(int(args.gpu))
    with open(args.config) as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        config[k] = v
    pprint(config)
    config = EasyDict(config)

    main(config)
