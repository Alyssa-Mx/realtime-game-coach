#!/usr/bin/env python3
"""验证 merge + BATCH=32 没改变输出：同一批帧，四种配置逐字段比对。
  base   未merge BATCH=16   ← 改动前的服务配置
  merge  merge  BATCH=16
  b32    merge  BATCH=32    ← 改动后的服务配置
只比字符串是否完全相同，不打分。用法: python train/verify_serving_change.py [帧数=20]
"""
import json, os, sys, torch, cv2, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('WEAPON_ONE_PROMPT', '1')
import tasks as T
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
MODEL=os.environ.get('BASE_MODEL', 'Qwen/Qwen3-VL-8B-Instruct')
ADAPTER=os.environ.get('ADAPTER','data/sft/v7/final')
N=int(sys.argv[1]) if len(sys.argv)>1 else 20
FIELDS=[k for k in T.GRID_TASKS if k!='supplies']+list(T.SUP_KEYS)
m=T.PRECISE['P8']
idx={r['game_t']:r['frame'] for r in map(json.loads,open(T.R+'/data/frames/%s/index.jsonl'%m,encoding='utf-8'))}
rows=[json.loads(l) for l in open(T.R+'/data/frames/%s/gamestate.jsonl'%m,encoding='utf-8') if json.loads(l)['game_t'] in idx]
rows=rows[::max(1,len(rows)//N)][:N]
frames=[cv2.imread(T.R+'/data/frames/%s/%s'%(m,idx[r['game_t']])) for r in rows]
proc=AutoProcessor.from_pretrained(MODEL); tok=proc.tokenizer; proc.tokenizer.padding_side='left'
def prompt(t):
    return proc.apply_chat_template([{'role':'system','content':T.SYSTEM},{'role':'user','content':[{'type':'image'},{'type':'text','text':t}]}],tokenize=False,add_generation_prompt=True)
def load(merge):
    b=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16).to('cuda:0')
    p=PeftModel.from_pretrained(b,ADAPTER if ADAPTER.startswith('/') else T.R+'/'+ADAPTER)
    return (p.merge_and_unload() if merge else p).eval()
def run(model,bs):
    out=[]
    for f in frames:
        ims=[T.crop(f,k,m,scale=T.SCALE_INFER.get(k,T.SCALE)) for k in FIELDS]; d={}
        for i in range(0,len(FIELDS),bs):
            ks=FIELDS[i:i+bs]
            enc=proc(text=[prompt(T.PROMPTS[k]) for k in ks],images=ims[i:i+bs],padding=True,return_tensors='pt').to('cuda:0')
            with torch.no_grad(): o=model.generate(**enc,max_new_tokens=192,do_sample=False)
            for j,k in enumerate(ks): d[k]=tok.decode(o[j][enc['input_ids'].shape[1]:],skip_special_tokens=True).strip()
        out.append(d)
    return out
print('帧数 %d，字段 %d，共 %d 条'%(len(frames),len(FIELDS),len(frames)*len(FIELDS)),flush=True)
mdl=load(False); base=run(mdl,16); del mdl; torch.cuda.empty_cache()
mdl=load(True);  mrg=run(mdl,16); b32=run(mdl,32); del mdl
def cmp(a,b,na,nb):
    """逐条比对；对每处差异，拿真值判定谁对 —— 只报字符串变没变没有意义。"""
    diff=[]; n=0
    for r,x,y in zip(rows,a,b):
        for k in FIELDS:
            n+=1
            if x[k]==y[k]: continue
            tg=T.target(r,k)
            if tg is None or tg[1]: sa=sb=None            # 无真值/被插补，不判定
            else:
                sa=T.score(k,T.parse_pred(k,x[k]) if hasattr(T,'parse_pred') else T.parse_json(x[k]),tg[0])
                sb=T.score(k,T.parse_pred(k,y[k]) if hasattr(T,'parse_pred') else T.parse_json(y[k]),tg[0])
                sa=sa.get('acc',sa.get('json_ok',0)); sb=sb.get('acc',sb.get('json_ok',0))
            diff.append((r['game_t'],k,x[k],y[k],sa,sb))
    print('\n%s vs %s: %d/%d 条不同 (%.2f%%)'%(na,nb,len(diff),n,100*len(diff)/n))
    wa=wb=tie=unk=0
    for gt,k,x,y,sa,sb in diff:
        if sa is None: unk+=1; v='无真值'
        elif sa>sb: wa+=1; v='%s 对'%na
        elif sb>sa: wb+=1; v='%s 对'%nb
        else: tie+=1; v='都%s'%('对' if sa==1 else '错')
        print('  t=%-7s %-14s %-34r -> %-34r  [%s]'%(gt,k,x[:32],y[:32],v))
    print('  判定: %s 更好 %d 次 | %s 更好 %d 次 | 打平 %d | 无真值 %d'%(na,wa,nb,wb,tie,unk))
    return len(diff)
cmp(base,mrg,'未merge','merge')
cmp(mrg,b32,'BATCH16','BATCH32')
cmp(base,b32,'改动前','改动后')
