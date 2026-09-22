"""Stage 0 零样本基线：裁剪 + prompt，Qwen3-VL-8B-Instruct 贪心生成，写 data/zeroshot/<tag>/s<shard>.jsonl。
用法: zeroshot.py <model_dir> <device> <shard> <nshards>
env: ADAPTER(LoRA 目录,评 SFT 用) TAG(输出子目录,默认 v1) MATCHES(P1,P2..; 默认 8 场全) TASKS(逗号分隔,默认 21 个) STRIDE(隔帧,默认 1) BATCH(默认 16)
改自 pipeline/gpu_vlm4_all.py（框/prompt/目标改从 train/tasks.py 取）。断点续跑：已写的 (match,game_t,task) 跳过。"""
import json, os, sys, glob, time, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T
MODEL, DEV, SHARD, NSH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
TAG = os.environ.get('TAG', 'v1'); BATCH = int(os.environ.get('BATCH', '16')); STRIDE = int(os.environ.get('STRIDE', '1'))
MS = [T.PRECISE[v] for v in os.environ.get('MATCHES', ','.join(T.PRECISE)).split(',')]
KEYS = os.environ.get('TASKS', ','.join(T.TASKS)).split(',')
OUTD = T.R + '/data/zeroshot/' + TAG; os.makedirs(OUTD, exist_ok=True)
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)   # 环境里没有 accelerate，不能用 device_map
if os.environ.get('ADAPTER'):                        # 评 SFT：挂 LoRA 并合并
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.environ['ADAPTER']).merge_and_unload()
proc = AutoProcessor.from_pretrained(MODEL); proc.tokenizer.padding_side = 'left'
print('model loaded', flush=True)
done = set()
for f in glob.glob(OUTD + '/s*.jsonl'):
    for l in open(f, encoding='utf-8'):
        r = json.loads(l); done.add((r['match'], r['game_t'], r['task']))
# 按帧分片（同一帧的 21 个任务留在一个分片里，帧只读一次）；先分片再滤 done（08-31 踩过反过来会错位）
todo, fi, last = [], -1, None
for s in T.iter_samples(MS, KEYS, STRIDE):
    if (s[0], s[1]) != last: last = (s[0], s[1]); fi += 1
    if fi % NSH == SHARD and (s[0], s[2], s[3]) not in done: todo.append(s)
print('todo', len(todo), flush=True)
out = open(OUTD + '/s%d.jsonl' % SHARD, 'a', encoding='utf-8')
# 一个 batch 里输出最长的样本决定整批解码步数（supplies ~100 token vs 多数 ~15），所以按"窗口内同任务成批"：
# 每 GROUP 个样本为一窗，窗内按任务排序，窗内的帧读一次缓存（09-03 实测混批 3.6/s）
GROUP = int(os.environ.get('GROUP', '1008'))
t0 = time.time(); n = 0
for g0 in range(0, len(todo), GROUP):
  win = sorted(todo[g0:g0 + GROUP], key=lambda s: (s[3], s[0], s[2])); fcache = {}
  for i in range(0, len(win), BATCH):
    chunk = win[i:i + BATCH]; imgs, texts = [], []
    for m, fr, gt, k, tgt, imp in chunk:
        if (m, fr) not in fcache: fcache[(m, fr)] = cv2.imread(T.R + '/data/frames/%s/%s' % (m, fr))
        imgs.append(T.crop(fcache[(m, fr)], k, m))
        msgs = [{'role': 'system', 'content': T.SYSTEM},
                {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': T.PROMPTS[k]}]}]
        texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    inputs = proc(text=texts, images=imgs, padding=True, return_tensors='pt').to(DEV)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    for (m, fr, gt, k, tgt, imp), ans in zip(chunk, proc.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)):
        out.write(json.dumps({'match': m, 'game_t': gt, 'task': k, 'target': tgt, 'imputed': imp, 'pred_raw': ans}, ensure_ascii=False) + '\n')
    out.flush(); n += len(chunk)
    if n % 800 < BATCH: print('%d/%d %.2f/s' % (n, len(todo), n / (time.time() - t0)), flush=True)
print('DONE', n, flush=True)
