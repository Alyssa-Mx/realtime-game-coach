#!/usr/bin/env python3
"""直接对比：整帧一次调用 vs 31 个裁剪块一次成批调用，量墙钟。

回答的问题：如果改成"一张大图 + 一条整合 prompt + 一个 JSON"，推理会不会更快？
两条成本都要看：
  视觉编码 —— 整帧 1280x720 x2 = 3600 token；裁剪块按 eval/scale_by_field.json 只有 846
  解码     —— 整帧是 1 条约 300 token 的序列；31 个块成批时墙钟取**最长的那条**（约 60），不是求和
只量时间，不看精度（v6 没训过整帧，整帧的输出必然是错的，但耗时是真的）。
用法: python train/bench_whole_vs_crops.py <frame.jpg> [match_id] [轮数=3]
"""
import json, os, sys, time, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('WEAPON_ONE_PROMPT', '1')
import tasks as T
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

MODEL = os.environ.get('BASE_MODEL', 'Qwen/Qwen3-VL-8B-Instruct')
ADAPTER = os.environ.get('ADAPTER', 'data/sft/v6/final')
FIELDS = [k for k in T.GRID_TASKS if k != 'supplies'] + list(T.SUP_KEYS)
proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer
proc.tokenizer.padding_side = 'left'
model = PeftModel.from_pretrained(
    AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to('cuda:0'),
    ADAPTER if ADAPTER.startswith('/') else T.R + '/' + ADAPTER).eval()

def prompt(text):
    return proc.apply_chat_template([{'role': 'system', 'content': T.SYSTEM},
                                     {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}],
                                    tokenize=False, add_generation_prompt=True)

path = sys.argv[1]; match = sys.argv[2] if len(sys.argv) > 2 else None
R = int(sys.argv[3]) if len(sys.argv) > 3 else 3
f = cv2.imread(path)

# 整帧：和 v3 的 S4 档一样，1280x720 最近邻 x2
whole = Image.fromarray(f[:, :, ::-1]); whole = whole.resize((whole.width * 2, whole.height * 2), Image.NEAREST)
WHOLE_PROMPT = ('这是《和平精英》一整帧游戏画面。读出 HUD 上的全部字段。\n输出 JSON：{'
                + ', '.join('"%s": ...' % k for k in T.GRID_TASKS) + '}\n看不清或不存在的用 null。')

def bench(name, fn):
    ts = []
    for _ in range(R):
        torch.cuda.synchronize(); t0 = time.time(); n_out = fn(); torch.cuda.synchronize(); ts.append((time.time() - t0, n_out))
    ts.sort(key=lambda x: x[0])
    md = ts[len(ts) // 2]
    print('%-26s %.3f s   生成 %d token   (%s)' % (name, md[0], md[1], ' '.join('%.2f' % t for t, _ in ts)), flush=True)
    return md[0]

def run_whole():
    enc = proc(text=[prompt(WHOLE_PROMPT)], images=[whole], return_tensors='pt').to('cuda:0')
    with torch.no_grad(): o = model.generate(**enc, max_new_tokens=512, do_sample=False)
    return int(o.shape[1] - enc['input_ids'].shape[1])

def run_crops(fast):
    tot = 0
    for i in range(0, len(FIELDS), 16):
        ks = FIELDS[i:i + 16]
        ims = [T.crop(f, k, match, scale=(T.SCALE_INFER.get(k, T.SCALE) if fast else T.SCALE)) for k in ks]
        enc = proc(text=[prompt(T.PROMPTS[k]) for k in ks], images=ims, padding=True, return_tensors='pt').to('cuda:0')
        with torch.no_grad(): o = model.generate(**enc, max_new_tokens=192, do_sample=False)
        tot += int(o.shape[1] - enc['input_ids'].shape[1])
    return tot

a = bench('整帧 1 次调用', run_whole)
b = bench('31 块 x3（旧）', lambda: run_crops(False))
c = bench('31 块 混合倍数（现在）', lambda: run_crops(True))
print('\n整帧 vs 现在的裁剪链路：%.2fx %s' % (a / c, '更慢' if a > c else '更快'))
