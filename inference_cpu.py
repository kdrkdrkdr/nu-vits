import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import sys
import re
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text = re.sub('[\s+]', ' ', text)
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


model_name = sys.argv[1]
model_step = sys.argv[2]



# Inference
hps = utils.get_hparams_from_file(f"./configs/{model_name}.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint(f"../vits_model/{model_name}/G_{model_step}.pth", net_g, None)

text_list = [
    '領域展開…無量恐妻',
    '先生、ちょっとお時間いただけますか？',
    '先生…一緒にライディングしに行こう。',
    '行ってきてからは一緒に… お風呂に入ろう。',
    '先生…どこ行くの…？ 今日寝て行くんじゃなかったの？拒否権はないよ。 …こっちおいで。',
    'お風呂にする…？ご飯にする…？それとも……あ…た……し？',
    '誰かに必要とされていることそれはあなたが誰かの希望であることだ',
    'しばらく焚火の跡をかき回してみたが、残念ながら灰の中には食べられそうなものは残っていなかった。',
    '先生…今日、隣で一緒に寝てもいい……？',
    '先生…私……下が…むずむずする……',
    '俺は南の洞窟に着くと、活発化している魔物を避けながら洞窟の最深部に進んだ。',
    '身体強化で反射神経や足の筋肉を強化しているから、魔物を避けるなんて楽勝。一々相手していたらキリがない。',
    'しかも村長の話では、村に被害を与えている訳でもない……',
    '聖女のみが使える聖魔法属性と膨大な魔力量を持って生まれたことで、二歳の頃に神殿へと引き取られた。',
    '今俺の目の前にはゴブリンの集落……',
    '洞窟の最深部は天井に穴が開いており、たくさんの草花が生えている美しい場所だ。',
]

# text_list = [
#     '내가 누군가를 좋아한다는 사실이, 그 사람에게는 상처가 될 수 있잖아요.',
#     '불편하면, 자세를 고쳐 앉아.',
#     '아무나 다 달에 가면, 지구에 남아서 버튼은 누가 눌러주냐고',
#     '원래, 다 보이면 노꼴이야.',
#     '천애고아들이 따로 없어요 그냥!',
#     '유연한 사고, 남탓하지 않기, 유연한 남탓, 사고하지 않기, 경직된 사고, 남탓. 우리 팀 개새끼',
#     '아니 남탓은 금지지만 팀은 남이 아니잖아! 팀은 이제 가족이니까 탓을 좀 해도 되지...',
#     '이길 수 있다면 구두 밑창이라도 핥겠다.',
#     '안심하세요. 도둑입니다. 별거 아니에요!',
#     '우주에서 바라 본 지구는 파란 유리 구슬 같았다.',
# ]


speed = 1
for idx, text in enumerate(text_list):
    sid = torch.LongTensor([idx])
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1/speed)[0][0,0].data.cpu().float().numpy()
    write(f'/content/drive/MyDrive/vits_output/{model_name}/{model_name}_{model_step}_{idx}.wav', hps.data.sampling_rate, audio)
    print(f'{model_name}_{model_step}_{idx} 음성 합성 완료')
    # break