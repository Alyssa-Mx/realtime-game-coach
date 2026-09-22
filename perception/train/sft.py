"""Stage 1 多任务 LoRA SFT（Qwen3-VL-8B-Instruct）。环境: transformers 4.57.6 + peft 0.20。
单卡: python train/sft.py <model_dir> <out_dir>     多卡: torchrun --nproc_per_node N train/sft.py <model_dir> <out_dir>
env: TRAIN(默认 P2,P3,P4,P6,P8) TASKS(默认全部) STRIDE(隔帧,默认 2) EPOCHS(1) LR(1e-4) R(16) FPB(每步帧数,默认 1)
     ACCUM(1) SAVE_EVERY(200 步) JITTER(裁框抖动 px,默认 3) SKIP_IMPUTED(1)
     SPEC: 多组混训，覆盖 TRAIN/TASKS/STRIDE。格式 `场次:任务:隔帧[:每项帧数]`，多组用 ; 分隔。
       场次: P1..P8 逗号列表 | coarse(54 场粗标) | all      任务: all | grid(21 整框) | cells(11 物资单格) | 逗号列表
       每项帧数>1 时把若干帧打成一个训练项（粗标只训武器时一帧只有 1-2 个裁块，不打包的话每步都在等读帧）。
       例: SPEC='P2,P3,P4,P6,P8:all:2 ; coarse:weapon_slot1,weapon_slot2:8:6'
一个"样本"=一帧的全部任务裁块（帧只读一次，读帧 0.45s/张是瓶颈）。labels 只算 assistant 段。
v3 计划模式（TRAIN_PLAN_v3.md §五）：PLAN=data/sft/v3/plan.jsonl 时不用 SPEC/DistributedSampler，按计划逐步走，
  rank r 取该步 ids[r]（粗标 ids[4r:4r+4]），用 multiscale.make_item/make_coarse 现场出块，总步数 = 计划行数。
  ADAPTER=<dir> 热启动（只载 adapter 权重，optimizer/scheduler 全新；目录里有 opt.pt 且 START_STEP>0 才恢复 optimizer）
  START_STEP=N 断点续跑（跳过计划前 N 步，scheduler 对齐）   PROBE=1 每步打 image_grid_thw / 视觉 token / 峰值显存 / 步时"""
import json, os, sys, time, math, random, collections, torch, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks as T, multiscale as M
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model, PeftModel
MODEL, OUT = sys.argv[1], sys.argv[2]
E = os.environ.get
TRAIN = [T.PRECISE[v] for v in E('TRAIN', ','.join(T.SPLIT['train'])).split(',')]
KEYS = E('TASKS', ','.join(T.TASKS)).split(','); STRIDE = int(E('STRIDE', '2')); EPOCHS = float(E('EPOCHS', '1'))
LR = float(E('LR', '1e-4')); RANK = int(E('R', '16')); FPB = int(E('FPB', '1')); ACCUM = int(E('ACCUM', '1'))
SAVE_EVERY = int(E('SAVE_EVERY', '200')); JITTER = int(E('JITTER', '3')); SKIP_IMP = E('SKIP_IMPUTED', '1') == '1'
SCALE_FIT = E('SCALE_FIT', '0') == '1'   # 按各字段推理倍数训练（消除 train/infer 失配）
PLAN = E('PLAN'); ADAPTER = E('ADAPTER'); START = int(E('START_STEP', '0')); PROBE = E('PROBE', '0') == '1'
dist = int(E('WORLD_SIZE', '1')) > 1
if dist: torch.distributed.init_process_group('nccl'); rank = torch.distributed.get_rank(); world = torch.distributed.get_world_size()
else: rank, world = 0, 1
dev = torch.device('cuda', int(E('LOCAL_RANK', '0'))); torch.cuda.set_device(dev)
log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

# ---- 数据：按帧组织；一个 item = 一帧(或一小包帧)的全部任务裁块 ----
def _ms(spec):
    spec = spec.strip()
    if spec == 'coarse': return list(T.COARSE)
    if spec == 'all': return list(T.PRECISE.values()) + list(T.COARSE)
    return [T.PRECISE.get(v.strip(), v.strip()) for v in spec.split(',')]
def _ks(spec):
    out = []
    for v in spec.split(','):
        v = v.strip()
        if v == 'all': out += list(T.TASKS)
        elif v == 'grid': out += list(T.GRID_TASKS)
        elif v == 'cells': out += list(T.SUP_KEYS)
        elif v: out.append(v)
    return list(dict.fromkeys(out))

GROUPS = [g.strip() for g in E('SPEC', '').split(';') if g.strip()] or \
         ['%s:%s:%d' % (','.join(E('TRAIN', ','.join(T.SPLIT['train'])).split(',')), ','.join(KEYS), STRIDE)]
items, stat = [], []
if PLAN:
    plan = [json.loads(l) for l in open(T.R + '/' + PLAN if not PLAN.startswith('/') else PLAN)]
    assert all(p['step'] == i + 1 for i, p in enumerate(plan))
    ROWS, IDX = {}, {}
    for m in {m for p in plan for m, _ in p['ids']}:
        ROWS[m] = {r['game_t']: r for r in map(json.loads, open(T.R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8'))}
        IDX[m] = {r['game_t']: r['frame'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
    assert world == 8 or E('ALLOW_WORLD'), 'plan.jsonl 按 8 rank 生成（ALLOW_WORLD=1 强行跑：rank 取 ids[r % 8]，只用于单卡调试）'
    plan = plan[START:]
    log('plan', PLAN, 'steps', len(plan) + START, 'start', START, collections.Counter((p['q'], p['s']) for p in plan))
for g in GROUPS if not PLAN else []:
    part = g.split(':'); ms, ks, st = _ms(part[0]), _ks(part[1]), int(part[2])
    bundle = int(part[3]) if len(part) > 3 else 1
    fr_map = {}
    for m, fr, gt, k, tgt, imp in T.iter_samples(ms, ks, st):
        if SKIP_IMP and imp: continue
        fr_map.setdefault((m, fr), []).append((k, tgt))
    fl = [(m, fr, lst) for (m, fr), lst in sorted(fr_map.items())]
    grouped = [fl[i:i + bundle] for i in range(0, len(fl), bundle)]
    items += grouped
    stat.append('%s → %d 帧 / %d 样本 / %d 项' % (part[0][:24], len(fl), sum(len(x[2]) for x in fl), len(grouped)))
if not PLAN:
    random.Random(0).shuffle(items)          # 各组混匀，别让粗标武器全挤在最后
    for x in stat: log('  spec', x)
    log('items', len(items), 'samples', sum(len(l) for it in items for _, _, l in it))
proc = AutoProcessor.from_pretrained(MODEL); tok = proc.tokenizer; tok.padding_side = 'right'

def _prompt(text, task=None):
    msgs = [{'role': 'system', 'content': T.system_for(task) if task else T.SYSTEM},
            {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def _pack(im, text, tgt, task=None):
    ans = T.ans_text(task, tgt) if task else json.dumps(tgt, ensure_ascii=False)
    ans += '<|im_end|>\n'
    return (im, _prompt(text, task) + ans, len(tok(ans, add_special_tokens=False).input_ids))

def _bright(im):
    if random.random() < 0.5:
        from PIL import ImageEnhance; im = ImageEnhance.Brightness(im).enhance(random.uniform(0.9, 1.1))
    return im

class DS(Dataset):
    def __len__(self): return len(items)
    def __getitem__(self, i):
        out = []
        for m, fr, lst in items[i]:
            f = cv2.imread(T.R + '/data/frames/%s/%s' % (m, fr))
            out += self._one(f, m, lst)
        return out
    def _one(self, f, m, lst):
        out = []
        for k, tgt in lst:
            im = T.crop(f, k, m, jitter=JITTER)
            if random.random() < 0.5:                       # 轻度增强：缩放 0.9–1.1 / 亮度 ±10%
                s = random.uniform(0.9, 1.1); im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))))
            out.append(_pack(_bright(im), T.PROMPTS[k], tgt))
        return out

class PlanDS(Dataset):
    """一条 = 计划的一步里本 rank 的那份：精标 1 帧 → 该档全部块；粗标 4 帧 → 武器块。增强只做亮度（缩放会破坏三档同像素密度的设定）。"""
    def __len__(self): return len(plan)
    def __getitem__(self, i):
        p = plan[i]; r = rank % 8; per = 4 if p['q'] == 'coarse' else 1; seed = p['crop_seed'] * 16 + r
        if 'fields' in p:                      # v4：只训 S1，字段由计划直接指定（每块一个字段）
            out = []
            for k, (m, gt) in enumerate(p['ids']):
                if k % 8 != r: continue        # 8 个 rank 轮流领 ids（coarse 一步 16 帧，每 rank 2 帧）
                f = cv2.imread(T.R + '/data/frames/%s/%s' % (m, IDX[m][gt])); row = ROWS[m][gt]
                for task in p['fields'][k]:
                    tg = T.target(row, task)
                    if tg is None: continue
                    # SCALE_FIT=1：按各字段实际推理用的放大倍数训练，消除 train/infer 失配，
                    # 顺带把视觉 token 降到约 1/4（2026-09-07 实测低倍数不掉分，见 eval/scale_by_field.json）
                    sc = T.SCALE_INFER.get(task, T.SCALE) if SCALE_FIT else T.SCALE
                    out.append(_pack(_bright(T.crop(f, task, m, jitter=min(JITTER, 1) if task.startswith('sup_') else JITTER, scale=sc)),
                                     T.PROMPTS[task], tg[0], task))
            return out
        out = []
        for j, (m, gt) in enumerate(p['ids'][r * per:(r + 1) * per]):
            f = cv2.imread(T.R + '/data/frames/%s/%s' % (m, IDX[m][gt])); row = ROWS[m][gt]
            it = M.make_coarse(f, m, row, p['s'], seed + j, jitter=JITTER) if p['q'] == 'coarse' else M.make_item(f, m, row, p['s'], seed, jitter=JITTER, skip_imputed=SKIP_IMP)
            out += [_pack(_bright(im), pr, ans) for im, pr, ans, meta in it]
        return out

def collate(batch):
    flat = [x for b in batch for x in b]
    imgs, texts, nans = zip(*flat)
    enc = proc(text=list(texts), images=list(imgs), padding=True, return_tensors='pt')
    labels = enc.input_ids.clone(); labels[:] = -100
    L = enc.attention_mask.sum(1)
    for i, n in enumerate(nans): labels[i, L[i] - n:L[i]] = enc.input_ids[i, L[i] - n:L[i]]
    enc['labels'] = labels
    return enc

if PLAN:   # 计划模式：每个 rank 顺序读同一份计划、各取各的 ids，不 shuffle、不用 DistributedSampler
    ds = PlanDS(); sampler = None; FPB = 1
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(E('WORKERS', '6')), collate_fn=collate, persistent_workers=True, prefetch_factor=4)
else:
    ds = DS()
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, seed=0) if dist else None
    dl = DataLoader(ds, batch_size=FPB, shuffle=sampler is None, sampler=sampler, num_workers=int(E('WORKERS', '6')),
                    collate_fn=collate, persistent_workers=True, prefetch_factor=4)

# ---- 模型 ----
model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.enable_input_require_grads()   # 非重入，DDP 下不会报 mark-ready-twice
if ADAPTER:   # 热启动：只载 adapter 权重（视觉塔、merger 都冻结不变），optimizer/scheduler 全新
    model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=True); log('warm start from', ADAPTER)
else:
    cfg = LoraConfig(r=RANK, lora_alpha=2 * RANK, lora_dropout=0.05, bias='none',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, cfg)
if rank == 0: model.print_trainable_parameters()
if dist: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index], find_unused_parameters=False)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
total = (len(plan) + START) if PLAN else int(math.ceil(len(dl) * EPOCHS / ACCUM)); warm = max(1, total // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
if START:
    for _ in range(START): sched.step()                       # scheduler 对齐到断点
    if ADAPTER and os.path.exists(ADAPTER + '/opt.pt'): opt.load_state_dict(torch.load(ADAPTER + '/opt.pt', map_location=dev)); log('optimizer resumed')
os.makedirs(OUT, exist_ok=True)
def save(tag):
    if rank != 0: return
    (model.module if dist else model).save_pretrained(OUT + '/' + tag); torch.save(opt.state_dict(), OUT + '/' + tag + '/opt.pt'); log('saved', tag)

# ---- 验证：留出场的固定样本，teacher-forced 出 val/loss 与答案 token 准确率；
#      每 EVAL_GEN_EVERY 步再贪心解码一小批，出字段级 val/acc（真正可比的那个数）。
VAL_PLAN = E('VAL_PLAN') or (os.path.dirname(PLAN) + '/val.jsonl' if PLAN else '')
if VAL_PLAN and not VAL_PLAN.startswith('/'): VAL_PLAN = T.R + '/' + VAL_PLAN
EVAL_EVERY = int(E('EVAL_EVERY', '100')); EVAL_GEN_EVERY = int(E('EVAL_GEN_EVERY', '400'))
GEN_N = int(E('EVAL_GEN_N', '256')); VB = int(E('EVAL_BS', '8'))
val_items = []
if VAL_PLAN and os.path.exists(VAL_PLAN):
    val_items = [json.loads(l) for l in open(VAL_PLAN, encoding='utf-8')]
    for it in val_items:
        m = it['match']
        if m not in ROWS:
            ROWS[m] = {r['game_t']: r for r in map(json.loads, open(T.R + '/data/frames/%s/gamestate.jsonl' % m, encoding='utf-8'))}
            IDX[m] = {r['game_t']: r['frame'] for r in map(json.loads, open(T.R + '/data/frames/%s/index.jsonl' % m, encoding='utf-8'))}
    val_items = val_items[rank::world]                        # 逐 rank 分片，最后 all_reduce
    log('val set %d 条（本 rank %d 条）来自 %s' % (len(val_items) * world, len(val_items), VAL_PLAN))
_vcache = {}
def _vblock(it):
    k = (it['match'], it['game_t'], it['task'])
    if k not in _vcache:
        f = cv2.imread(T.R + '/data/frames/%s/%s' % (it['match'], IDX[it['match']][it['game_t']]))
        row = ROWS[it['match']][it['game_t']]
        _vcache[k] = (T.crop(f, it['task'], it['match']), T.PROMPTS[it['task']], T.target(row, it['task'])[0])
    return _vcache[k]

TASK_IX = {k: i for i, k in enumerate(T.TASKS)}; NT = len(T.TASKS)
def validate(do_gen):
    model.eval(); per_task = {}
    tl = torch.zeros(4, device=dev)                            # loss*n, n, 对的token, 总token
    with torch.no_grad():
        for i in range(0, len(val_items), VB):
            its0 = val_items[i:i + VB]; chunk = [_vblock(it) for it in its0]
            enc = collate([[_pack(im, pr, tg, it['task']) for (im, pr, tg), it in zip(chunk, its0)]])
            enc = {k: v.to(dev) for k, v in enc.items()}
            out = model(**enc)
            m = enc['labels'] != -100
            tl[0] += out.loss.item() * int(m.sum()); tl[1] += int(m.sum())
            pr = out.logits[:, :-1].argmax(-1); gt = enc['labels'][:, 1:]; mm = gt != -100
            tl[2] += int(((pr == gt) & mm).sum()); tl[3] += int(mm.sum())
    acc_field = -1.0
    if do_gen:
        hit = torch.zeros(2, device=dev); per = torch.zeros(2 * NT, device=dev)
        gm = (model.module if dist else model)
        # 三件必须做对的事（2026-09-04 第一版全踩了，val/acc_field 假低到 0.498）：
        # 1) 批量 generate 必须 **左 padding**：默认右 padding 时，短序列的续写从 pad 之后开始，
        #    按 input_ids.shape[1] 切出来的就是垃圾 —— 一批 8 条里只有最长的那条是对的。
        # 2) max_new_tokens 要够：supplies 的答案 147 字符（约 80 token），48 会被截断，永远判错。
        # 3) 取样要在整个验证集上均匀跨步，不能取前 N 条 —— 前 N 条只覆盖开局那几帧。
        side = proc.tokenizer.padding_side; proc.tokenizer.padding_side = 'left'
        stride = max(1, len(val_items) // max(GEN_N, 1))
        picks = val_items[::stride][:GEN_N]
        for i in range(0, len(picks), VB):
            its = picks[i:i + VB]; chunk = [_vblock(it) for it in its]
            enc = proc(text=[_prompt(pr, it['task']) for (_, pr, _), it in zip(chunk, its)],
                       images=[im for im, _, _ in chunk], padding=True, return_tensors='pt')
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                o = gm.generate(**enc, max_new_tokens=192, do_sample=False)
            for j, it in enumerate(its):
                txt = tok.decode(o[j][enc['input_ids'].shape[1]:], skip_special_tokens=True)
                sc = T.score(it['task'], T.parse_pred(it['task'], txt), chunk[j][2])
                a_ = sc.get('acc', sc.get('json_ok', 0.0))
                hit[0] += a_; hit[1] += 1
                ti = TASK_IX[it['task']]; per[ti] += a_; per[ti + NT] += 1
        proc.tokenizer.padding_side = side
        if dist: torch.distributed.all_reduce(hit); torch.distributed.all_reduce(per)
        acc_field = float(hit[0] / max(hit[1], 1))
        per_task = {k: (float(per[i] / per[i + NT]), int(per[i + NT])) for k, i in TASK_IX.items() if per[i + NT] > 0}
    if dist: torch.distributed.all_reduce(tl)
    model.train()
    return float(tl[0] / max(tl[1], 1)), float(tl[2] / max(tl[3], 1)), acc_field, per_task

step, t0, run = START, time.time(), []
model.train()
for ep in range(int(math.ceil(EPOCHS))):
    if sampler: sampler.set_epoch(ep)
    for i, batch in enumerate(dl):
        if step >= total: break
        if i == 0:   # 自检：labels 段解码回来必须正好是 assistant 的 JSON
            lb = batch['labels'][0]; log('label_check', repr(tok.decode(lb[lb != -100])))
        batch = {k: v.to(dev) for k, v in batch.items()}
        ts = time.time(); loss = model(**batch).loss / ACCUM; loss.backward(); run.append(loss.item() * ACCUM)
        if (i + 1) % ACCUM: continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        if PROBE:   # 每步：档 / 块数 / 视觉 token（grid 乘积 / merge²）/ 序列长 / 峰值显存 / 步时 —— A5 门槛看 S4 的 vis_tok
            thw = batch['image_grid_thw']; vt = int((thw[:, 0] * thw[:, 1] * thw[:, 2]).sum()) // 4
            p = plan[i] if PLAN else {}
            print('probe rank:%d step:%d %s/%s blocks:%d vis_tok:%d max_thw:%s seq:%d ans_tok:%d loss:%.4f mem_gb:%.1f dt:%.2f' % (
                rank, step, p.get('q'), p.get('s'), thw.shape[0], vt, thw.max(0).values.tolist(), batch['input_ids'].shape[1],
                int((batch['labels'] != -100).sum()), run[-1], torch.cuda.max_memory_allocated() / 1e9, time.time() - ts), flush=True)
        vs = ''
        if val_items and step % EVAL_EVERY == 0:
            # 必须在下面那行日志之前算完、并进同一行：外部日志解析脚本对同一个 step 只收第一行
            # (`if step <= last_step: continue`)，val 单独起一行会被静默丢掉 —— 2026-09-04 踩过。
            vl, vat, vaf, vpt = validate(step % EVAL_GEN_EVERY == 0)
            vs = ' - val/loss:%.4f - val/acc_tok:%.4f%s' % (vl, vat, '' if vaf < 0 else ' - val/acc_field:%.4f' % vaf)
            # 逐字段 acc：决定下一轮把标注/数据力气花在哪，比总均值有用
            vs += ''.join(' - val/f/%s:%.4f' % (k, v) for k, (v, n) in sorted(vpt.items()))
            if vpt: log('val per-task n: ' + ' '.join('%s=%d' % (k, n) for k, (_, n) in sorted(vpt.items())))
        if step % 10 == 0:
            # 行格式对齐 verl console（`step:N - k:v - ...`），外部脚本可原样解析后推 SwanLab
            log('step:%d - train/loss:%.4f - train/lr:%.2e - train/elapsed_s:%d - train/mem_gb:%.1f - train/total_steps:%d%s' % (
                step, sum(run[-10 * ACCUM:]) / len(run[-10 * ACCUM:]), sched.get_last_lr()[0], time.time() - t0,
                torch.cuda.max_memory_allocated() / 1e9, total, vs))
        if step % SAVE_EVERY == 0: save('step%d' % step)
save('final'); log('TRAIN_DONE', step)
