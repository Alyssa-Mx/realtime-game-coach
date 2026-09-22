"""v3 配对多尺度评测：同一批帧，S1/S2/S3/S4 四档全测（TRAIN_PLAN_v3.md §八）。随机块用固定种子（CROP_SEED），跨 checkpoint 可比。
用法: eval_v3.py <model_dir> <device> <shard> <nshards>
env: ADAPTER  TAG(输出 data/eval_v3/<tag>)  MATCHES(默认 P7)  STRIDE(4)  TIERS(S1,S2,S3,S4)  CROP_SEED(0)  BATCH(16)
     LATENCY=1: 不分片、只取前 LAT_N(20) 帧，每档按"整帧一次调用（同帧块成批）"和"逐块串行调用"各计一次墙钟，写 latency.jsonl
输出 s<shard>.jsonl 两种行：
  {kind:'field', tier, match, game_t, task, target, imputed, pred_raw(该字段反解成单字段 dict 的 JSON 或 null), block}  — score_v3.py 用 tasks.score 打分
  {kind:'block', tier, match, game_t, fields, json_ok, missing, extra, n_tok}"""
import json, os, sys, glob, time, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T, multiscale as M
MODEL, DEV, SHARD, NSH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
E = os.environ.get
TAG = E('TAG', 'v3'); BATCH = int(E('BATCH', '16')); STRIDE = int(E('STRIDE', '4')); SEED = int(E('CROP_SEED', '0'))
TIERS = E('TIERS', 'S1,S2,S3,S4').split(','); LAT = E('LATENCY', '0') == '1'; LAT_N = int(E('LAT_N', '20'))
# SCALE_FIT=1: S1 档按 eval/scale_by_field.json 的逐字段倍数裁剪（v7 起是这么训的）。默认关，保持与 v2~v6 评测同尺。
SCALE_FIT = E('SCALE_FIT', '0') == '1'
_sc = (lambda k: T.SCALE_INFER.get(k, T.SCALE)) if SCALE_FIT else (lambda k: None)
MS = [T.PRECISE[v] for v in E('MATCHES', 'P7').split(',')]
OUTD = T.R + '/data/eval_v3/' + TAG; os.makedirs(OUTD, exist_ok=True)
from transformers import AutoModelForImageTextToText, AutoProcessor
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
if E('ADAPTER'):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, E('ADAPTER')).merge_and_unload()
proc = AutoProcessor.from_pretrained(MODEL); proc.tokenizer.padding_side = 'left'
print('model loaded', flush=True)

def frames():
    fi = -1
    for m in MS:
        idx = {r['game_t']: r['frame'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
        for i, l in enumerate(open(T.R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8')):
            if i % STRIDE: continue
            r = json.loads(l)
            if r['game_t'] not in idx: continue
            fi += 1
            if LAT:
                if fi >= LAT_N: return
            elif fi % NSH != SHARD: continue
            yield m, idx[r['game_t']], r

def blocks_of(f, m, row, tier):
    """一帧一档的全部块：[(PIL, prompt, fields, targets{k: tgt})]。S1 = 21 整框各一块（不含物资单格；跨档只比 21 字段）。"""
    tg = M.avail(row)
    if tier == 'S1':
        return [(T.crop(f, k, m, scale=_sc(k)), T.PROMPTS[k], [k], {k: tg[k]}) for k in tg]
    seed = M.frame_hash(m, row['t_video'], SEED) % (1 << 30)
    return [(im, pr, meta['fields'], {k: tg[k] for k in meta['fields']}) for im, pr, ans, meta in M.make_item(f, m, row, tier, seed, jitter=0)]

def gen(imgs, texts):
    msgs = [[{'role': 'system', 'content': T.SYSTEM}, {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': t}]}] for t in texts]
    texts = [proc.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in msgs]
    inputs = proc(text=texts, images=imgs, padding=True, return_tensors='pt').to(DEV)
    with torch.no_grad(): ids = model.generate(**inputs, max_new_tokens=1200 if max(len(t) for t in texts) > 1500 else 400, do_sample=False)
    outs = ids[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(outs, skip_special_tokens=True), [int((o != proc.tokenizer.pad_token_id).sum()) for o in outs]

def emit(out, tier, m, row, fields, tgs, raw, ntok, blk):
    pred = T.parse_json(raw); ok = isinstance(pred, dict)
    for k in fields:
        if tier == 'S1': pr = raw                                            # 单字段块：整段原样交给 tasks.score（与 zeroshot 一致）
        else: pr = json.dumps(M.unval(pred[k], tgs[k]), ensure_ascii=False) if ok and k in pred else None
        out.write(json.dumps({'kind': 'field', 'tier': tier, 'match': m, 'game_t': row['game_t'], 'task': k, 'target': tgs[k], 'imputed': False, 'pred_raw': pr, 'block': blk}, ensure_ascii=False) + '\n')
    out.write(json.dumps({'kind': 'block', 'tier': tier, 'match': m, 'game_t': row['game_t'], 'fields': fields, 'json_ok': int(ok),
                          'missing': [k for k in fields if not (ok and k in pred)] if tier != 'S1' else [], 'extra': [k for k in pred if k not in fields] if ok and tier != 'S1' else [],
                          'n_tok': ntok, 'raw': raw if not ok else None}, ensure_ascii=False) + '\n')

if LAT:   # 延迟：单卡、逐帧，每档两种调用方式
    out = open(OUTD + '/latency.jsonl', 'a', encoding='utf-8')
    for m, fr, row in frames():
        f = cv2.imread(T.R + '/data/frames/%s/%s' % (m, fr))
        for tier in TIERS:
            bl = blocks_of(f, m, row, tier); rec = {'tier': tier, 'match': m, 'game_t': row['game_t'], 'blocks': len(bl)}
            torch.cuda.synchronize(); t = time.time(); raws, nt = gen([b[0] for b in bl], [b[1] for b in bl]); torch.cuda.synchronize(); rec['batched_s'] = time.time() - t
            t = time.time()
            for b in bl: gen([b[0]], [b[1]])
            torch.cuda.synchronize(); rec['sequential_s'] = time.time() - t; rec['out_tok'] = sum(nt)
            out.write(json.dumps(rec) + '\n'); out.flush(); print(rec, flush=True)
    print('DONE latency', flush=True); sys.exit()

done = set()
for fn in glob.glob(OUTD + '/s*.jsonl'):
    for l in open(fn, encoding='utf-8'):
        r = json.loads(l)
        if r['kind'] == 'block': done.add((r['tier'], r['match'], r['game_t']))
out = open(OUTD + '/s%d.jsonl' % SHARD, 'a', encoding='utf-8')
t0 = time.time(); n = 0
for tier in TIERS:
    todo = [(m, fr, row) for m, fr, row in frames() if (tier, m, row['game_t']) not in done]
    print(tier, 'frames', len(todo), flush=True)
    q = []          # 按块成批：同档块的 token 量相近；帧读一次
    def flush():
        raws, nts = gen([x[3] for x in q], [x[4] for x in q])
        for (m, row, blk, im, pr, fields, tgs), raw, nt in zip(q, raws, nts): emit(out, tier, m, row, fields, tgs, raw, nt, blk)
        out.flush(); q.clear()
    for m, fr, row in todo:
        f = cv2.imread(T.R + '/data/frames/%s/%s' % (m, fr))
        for blk, (im, pr, fields, tgs) in enumerate(blocks_of(f, m, row, tier)):
            q.append((m, row, blk, im, pr, fields, tgs))
            if len(q) >= (BATCH if tier != 'S4' else max(1, BATCH // 8)): flush()
        n += 1
        if n % 20 == 0: print('%s %d frames %.1fs' % (tier, n, time.time() - t0), flush=True)
    if q: flush()
print('DONE', n, flush=True)
