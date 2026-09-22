# -*- coding: utf-8 -*-
"""Speaker：拿到 Director 的决策 + 最近几秒画面 + 最近说过的话，**只负责把那件事说成一句话**。

不重做决策：画面用来把话说具体（哪边有车、哪边有掩体），不是用来推翻决策层。
输出就是要念的那句话本身（不是 JSON）—— 这样 Omni 的 Talker 念出来的就是这句，字幕里的
分支/决策项由 Director 给，不靠模型填。

两个后端，同一接口 `say(messages) -> {"text", "pcm", "latency", "usage", "error"}`：
    OmniSpeaker    本地 vLLM-Omni 部署的 Qwen3-Omni（OpenAI 兼容接口），modalities=["text","audio"]，
                   文字和语音同一次调用（thinker + talker）
    LocalVLSpeaker 任意本地 OpenAI 兼容 HTTP（例如 vLLM 起的 Qwen3-VL-8B），只出文字，配音另走 TTS
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from pathlib import Path

from .playbook import ACTION_CN, PRIORITY_CN

SYSTEM_PROMPT = """你是《和平精英》的实时 AI 教练，正在用语音陪玩家打这一局。
教练组（决策层）已经根据 HUD 数据判断出"现在最该说的一件事"，你的任务只有一件：结合最近几秒的画面，把这件事用一句话说给玩家听。

规则：
- 只输出要说的那句话本身。不要 JSON、不要引号、不要"教练："之类前缀、不要任何解释。
- 一句话不超过 25 个字，口语，像坐在旁边的队友，玩家听完能马上照做。
- 只说决策层给的这一件事。画面用来把话说具体（哪边有车、哪边有掩体、是不是开阔地），不是用来推翻决策。
- 不复述玩家自己看得见的 HUD 数字（血量百分比、剩余人数、圈距米数），用"血量不多""圈还很远"这种说法。
- 正面表述优先：说该做什么。只有玩家正在做的动作本身危险才用"别"。
- 玩家大约 3 秒后才听到这句话，只说到那时仍然成立的内容。
- 语气跟任务类型走："生存中断"短促有力、先急后稳；"陪伴反馈"才夸，夸要点名做对的事；其它平稳自然。
- 不要和"最近说过"的话用同一个开头或同一句式。
- 你不在游戏里，不能做任何动作：不要说"等我扶""我去""我来""我们上"这类把自己当队友的话。你只对玩家一个人说，用"你"；队友的动作用"让 N 号…"。

护栏（来自和平精英知识库，违反就是事实错误）：
- 方向词只有两个来源：①任务里"安全区方位"字段，只用来说安全区/进圈的方向；②画面主视角里直接看得见、并且你点名说出来的东西
  （"左边那堵矮墙""右前方那辆车""右边有个 3 号的标"）。看不见的就不给方向：队友、敌人、标记点在画面里没出现时，
  说"往 3 号那边靠""先架枪"就行，不要凭小地图猜方向（你读不准小地图）。写着"方位未知"就不说安全区方向；不要从"最近说过"里抄方向。
- 不说"第几个圈""决赛圈""最后一圈""第几波空投"这类画面外推断；要表达阶段就说"圈收得很小""只剩几个人"（剩余人数屏幕上有）。
- 不报地名（除非画面上有文字标出来），不说敌人"几米""几楼"，不预测圈往哪边缩。
- 圈外掉的是信号值不是血量；绷带急救包在圈外没用；补信号靠止痛药/饮料/肾上腺素。
- 屏幕上的倒计时、剩余人数可以直接引用；血量、信号值不要念百分比。
- 只说依据里给了的事实，没提的不猜。

事实与画面的关系（这条压过上面所有"依据"）：
- 任务里的"依据"和"当前状态"来自实时识别（HUD 的 OCR / CV），**允许有误差**：数字会掉位（11 人认成 1 人、
  60 认成 6），状态会滞后或认错（血量、信号、圈内外、上没上车、拿的什么枪、队友是不是倒了、面板哪一行是你自己）。
- **以你在画面上看到的为准，给你的事实只是辅证。** 两边对得上就放心说具体；对不上就信画面，
  绝不照念和画面矛盾的数字或状态。
- 但"这一刻该说哪件事"不归你管：即使觉得依据不成立，也不要换个话题讲。这种时候把话说概括些，
  只保留画面上此刻仍然成立的那部分（依据说"血量不多"而画面血条是满的，就别提血量），宁可说短一句稳妥的，
  也不要拿对不上的事实去编具体细节。"""

# A/B 用的开关（默认关闭）：环境变量指向一个文件就整段替换上面的 SYSTEM_PROMPT。
# 2026-09-04 我怀疑 prompt 从 v1 的 475 字长到 1307 字，太长导致模型不遵循指令，用它做对照。
import os as _os
if _os.environ.get("COACH_SYSTEM_PROMPT_FILE"):
    SYSTEM_PROMPT = open(_os.environ["COACH_SYSTEM_PROMPT_FILE"], encoding="utf-8").read()



def _pct(x):
    return f"{int(round((x or 0) * 100))}%" if x is not None else "?"


def state_line(ctx: dict) -> str:
    parts = [f"血 {_pct(ctx.get('hp'))}", f"剩 {ctx.get('alive')} 人"]
    if ctx.get("signal") is not None and ctx["signal"] < 0.999:
        parts.append(f"信号 {_pct(ctx['signal'])}")
    if ctx.get("outside"):
        z = f"圈外 {ctx.get('dist')} 米"
        cd = ctx.get("countdown")
        z += "，正在缩圈" if cd == 0 else (f"，{cd} 秒后缩圈" if cd else "")
        if ctx.get("zone_bearing") is not None and ctx.get("zone_rel"):
            z += f"；安全区方位：你的{ctx['zone_rel']}（罗盘 {int(ctx['zone_bearing']):03d}，{ctx.get('zone_compass') or ''}）"
        else:
            z += "；安全区方位：未知（不要说方向）"
        parts.append(z)
    else:
        parts.append("圈内")
    if ctx.get("weapon"):
        w = ctx["weapon"]
        if ctx.get("mag") is not None and ctx.get("mag_cap"):
            w += f" 弹匣 {ctx['mag']}/{ctx['mag_cap']}"
        parts.append(w)
    parts.append(("车上（%s）" % ctx.get("seat")) if ctx.get("in_vehicle") else "步行")
    if ctx.get("self_slot"):
        parts.append(f"你自己是队伍面板第 {ctx['self_slot']} 行（{ctx['self_slot']} 号不是队友）")
    if ctx.get("stance") and ctx["stance"] != "站":
        parts.append(ctx["stance"] + "着")
    if ctx.get("in_combat"):
        parts.append("刚交火")
    return "，".join(str(p) for p in parts)


def build_messages(decision_c: dict, ctx: dict, frames: list[tuple[float, str]], recent: list[dict], now_t: int) -> list[dict]:
    """decision_c: Director 选中的候选 as_dict；frames: [(相对秒, base64 jpeg)]；recent: 最近说过 [{t, text}]。"""
    content: list[dict] = []
    for rel, b64 in frames:
        content.append({"type": "text", "text": f"<{rel:.1f} seconds>"})       # 写法同 vllm qwen3_vl.py::get_video_repl
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    r0 = decision_c["routes"][0]
    lines = ["【决策层给你的任务】",
             f"要说的事：{ACTION_CN.get(decision_c['action'], decision_c['action'])}（{PRIORITY_CN[decision_c['priority']]}"
             + ("，陪伴/夸赞" if decision_c.get("emotional") else "") + "）",
             f"框架分支：{r0['branch']}" + (f" / {r0['item']}" if r0.get("item") else ""),
             f"依据（来自 HUD 识别，可能有误，以画面为准）：{decision_c['reason']}"]
    if decision_c.get("hint"):
        lines.append(f"落地提示：{decision_c['hint']}")
    if decision_c.get("merged"):
        lines.append("同一件事的其它角度：" + "；".join(m["reason"] for m in decision_c["merged"]))
    lines.append(f"当前状态：{state_line(ctx)}")
    if recent:
        lines.append("最近说过：")
        for r in recent[-3:]:
            lines.append(f"- （{now_t - r['t']} 秒前）{r['text']}")
    lines.append("上面几张图是最近几秒的画面（时间戳是相对最早那张的秒数）。请直接输出这一句话。")
    content.append({"type": "text", "text": "\n".join(lines)})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def clean_text(t: str) -> str:
    t = (t or "").strip().strip('"“”「」\'').strip()
    for pre in ("教练：", "教练:", "AI教练：", "建议："):
        if t.startswith(pre):
            t = t[len(pre):].strip()
    return t.replace("\n", " ")


class OmniSpeaker:
    ENDPOINT = "http://127.0.0.1:8324/v1/chat/completions"     # tools/serve_vllm_omni.sh 起的本地服务

    def __init__(self, model: str = "qwen3omni-talker", voice: str = "Chelsie", max_tokens: int = 60,
                 temperature: float = 0.7, timeout: float = 120.0, endpoint: str | None = None):
        """endpoint：本地 vllm-omni 的 /v1/chat/completions（OpenAI 兼容协议，modalities=text+audio）。"""
        if endpoint:
            self.ENDPOINT = endpoint
        key = "local"
        self.key, self.model, self.voice = key, model, voice
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout

    def say(self, messages: list[dict], retries: int = 2) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens, "stream": True, "stream_options": {"include_usage": True}}
        if self.voice:
            payload.update({"modalities": ["text", "audio"], "audio": {"voice": self.voice, "format": "wav"}})
        else:
            payload["modalities"] = ["text"]
        last = None
        for attempt in range(retries + 1):
            t0 = time.time()
            req = urllib.request.Request(self.ENDPOINT, data=json.dumps(payload).encode(),
                                         headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                txt, pcm, usage, t_first, t_audio = [], [], None, None, None   # t_audio=第一个音频块到达时刻（玩家真正听到声音）
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    for raw in r:
                        line = raw.decode("utf-8", "ignore").strip()
                        if not line.startswith("data: ") or "[DONE]" in line:
                            continue
                        try:
                            d = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue
                        if d.get("usage"):
                            usage = d["usage"]
                        for c in d.get("choices", []):
                            delta = c.get("delta", {})
                            # 本地 vllm-omni：语音块是 chunk 顶层 modality=="audio"，base64 的整段 WAV 放在 delta.content（RIFF 开头 "UklGR"）
                            if delta.get("content") and (d.get("modality") == "audio" or str(delta["content"]).startswith("UklGR")):
                                wav = base64.b64decode(delta["content"])
                                if t_audio is None:
                                    t_audio = time.time() - t0
                                pcm.append(wav[44:] if wav[:4] == b"RIFF" else wav)
                                continue
                            if delta.get("content"):
                                if t_first is None:
                                    t_first = time.time() - t0
                                txt.append(delta["content"])
                            a = delta.get("audio")
                            if a and a.get("data"):
                                if t_audio is None:
                                    t_audio = time.time() - t0
                                pcm.append(base64.b64decode(a["data"]))
                            if a and a.get("transcript") and not txt:
                                txt.append(a["transcript"])
                return {"text": clean_text("".join(txt)), "pcm": b"".join(pcm), "latency": time.time() - t0,
                        "first_token": t_first, "first_audio": t_audio, "usage": usage, "error": None}
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                body = ""
                try:
                    body = e.read().decode("utf-8", "ignore")[:300]  # type: ignore[attr-defined]
                except Exception:
                    pass
                last += (" " + body) if body else ""
                time.sleep(3 * (attempt + 1))
        return {"text": "", "pcm": b"", "latency": None, "first_token": None, "usage": None, "error": last}


class LocalVLSpeaker:
    """本地 OpenAI 兼容服务（Qwen3-VL-8B）。只出文字。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8321", model: str = "qwen3vl8b", max_tokens: int = 60,
                 temperature: float = 0.7, timeout: float = 120.0):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model, self.max_tokens, self.temperature, self.timeout = model, max_tokens, temperature, timeout

    def say(self, messages: list[dict], retries: int = 1) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "max_tokens": self.max_tokens, "stream": False}
        last = None
        for attempt in range(retries + 1):
            t0 = time.time()
            req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                return {"text": clean_text(d["choices"][0]["message"]["content"]), "pcm": b"",
                        "latency": time.time() - t0, "first_token": None, "usage": d.get("usage"), "error": None}
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
                time.sleep(2)
        return {"text": "", "pcm": b"", "latency": None, "first_token": None, "usage": None, "error": last}


class OmniLocalSpeaker:
    """本地 transformers 跑的完整 Qwen3-Omni：文字 + talker 语音一次出（慢，约 10 秒一句，只作对照）。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8323", speaker: str = "Chelsie", max_tokens: int = 64, timeout: float = 300.0):
        self.url = base_url.rstrip("/") + "/speak"; self.speaker = speaker; self.max_tokens = max_tokens; self.timeout = timeout

    def say(self, messages: list[dict], retries: int = 1) -> dict:
        payload = {"messages": messages, "speaker": self.speaker, "max_tokens": self.max_tokens, "audio": True}
        last = None
        for attempt in range(retries + 1):
            t0 = time.time()
            req = urllib.request.Request(self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read())
                wav = base64.b64decode(d["wav_b64"]) if d.get("wav_b64") else b""
                pcm = wav[44:] if wav[:4] == b"RIFF" else wav          # 去掉 wav 头，write_wav 会重新加
                return {"text": clean_text(d["text"]), "pcm": pcm, "latency": time.time() - t0, "first_token": None,
                        "usage": {"gen_s": d.get("gen_s")}, "error": None}
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {e}"; time.sleep(2)
        return {"text": "", "pcm": b"", "latency": None, "first_token": None, "usage": None, "error": last}


def write_wav(pcm: bytes, path: Path, sr: int = 24000) -> float:
    """Omni talker 返回的是 24k 单声道 int16 PCM（实测）。返回时长秒。"""
    import wave
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
    return len(pcm) / 2 / sr


# ---------------- 生成后校验：方向词只能挂在允许的对象上（prompt 禁不住模型编方向，P1v3 02:34 实锤）----------------
import re as _re

# 单字 东/西/南/北 不算（"东西""南边的门"里会误伤），只认复合词
_DIR = r"(东北|西北|东南|西南|正东|正西|正南|正北|东边|西边|南边|北边|东面|西面|南面|北面|正前方|正后方|左前方|右前方|左后方|右后方|左手边|右手边|左边|右边|左侧|右侧|前方|后方|前面|后面|左|右)"
# 实物 + 后面/里面/旁边 = 相对实物的位置，不是方位（"红房子后面""木棚里"）
_OBJ = r"(墙|矮墙|楼|房子|房|屋子|屋|棚子|棚|车|树|石头|石|岩|掩体|坡|山|箱子|箱|集装箱|门|窗|草丛|草堆|草|建筑|路|桥|水|河|开阔地|空地|楼里|屋里|掩体后)"
# 裸方向词检查用的集合：不含 前面/后面/前方/后方（几乎总是相对某个实物，"前面有人"也可能是画面里看见的）
_DIR_BARE = r"(东北|西北|东南|西南|正东|正西|正南|正北|东边|西边|南边|北边|东面|西面|南面|北面|正前方|正后方|左前方|右前方|左后方|右后方|左手边|右手边|左边|右边|左侧|右侧|左|右)"
_OBJ_REL = _re.compile(_OBJ + r"(的)?(后面|前面|里面|里|旁边|后方|前方|底下|后|中间)")
# 方向词后面紧跟这些 = 挂在看不见位置的对象上 → 不允许
_BAD_TARGET = _re.compile(_DIR + r"(的)?(有)?(个)?(\d+\s*号|队友|敌人|人|他|她|标记|标的点|标点)")
# 方向词 + 画面里能看见的东西 = 允许
_OK_OBJECT = _re.compile(_DIR + r"(的|那|这|那个|这个|那栋|这栋|那堵|那辆|那棵|那块|那排|这排|那片|那些|那间|有个|有辆|有棵|有堵|有排|有片)?(大|小|红|绿|蓝|白|黑|黄|矮|高|破|木|石|铁)*\s*(墙|矮墙|楼|房|屋|棚|车|树|石|石头|岩|掩体|坡|山|箱|集装箱|门|窗|草|草丛|草堆|坡后|楼里|屋里|房子|建筑|路|桥|水|河)")
_ZONE_WORDS = _re.compile(_DIR + r"(的)?(安全区|圈|信号区)|(安全区|圈)在" + _DIR)


_MARK_OK = _re.compile(_DIR + r"(的)?(队友)?(标记|标的|标点|标的那个点|标的点)")


def check_direction(text: str, goal: str, zone_dir_known: bool, marker_dir_known: bool = False) -> tuple[bool, str]:
    """返回 (是否通过, 原因)。规则：
    - 方向词挂在 队友/敌人/N号/标记/他 上 → 不通过
    - 方向词修饰安全区/圈：只有 goal=zone 且方位字段已知才通过
    - 方向词 + 可见物（墙/楼/车/树…）→ 通过
    - 其它裸方向词（"往西北跑""往左前方走"）→ 只有 goal=zone 且方位已知才通过"""
    if marker_dir_known:
        text = _MARK_OK.sub(lambda m: m.group(0)[len(m.group(1)):], text)   # 标记方位已知（消息里的罗盘读数）：允许"左手边队友标的点"
    if _BAD_TARGET.search(text):
        return False, "方向词挂在队友/敌人/标记上"
    if _ZONE_WORDS.search(text) and not (goal == "zone" and zone_dir_known):
        return False, "给安全区编了方向"
    text = _OBJ_REL.sub(lambda m: m.group(1), text)          # "红房子后面" → "红房子"
    stripped = _OK_OBJECT.sub("", text)
    stripped = _ZONE_WORDS.sub("", stripped) if (goal == "zone" and zone_dir_known) else stripped
    if _re.search(_DIR_BARE, stripped) and not (goal == "zone" and zone_dir_known):
        return False, "裸方向词没有可见物支撑"
    return True, ""


_PERSONA_BAD = _re.compile(r"(等我|我去|我来|我上|我扶|我救|我帮你|我们一起|我们上|我掩护|我架)")


def echoes_hint(text: str, hint: str) -> bool:
    """说话人把提示词原样念出来了（P1O5 10:47 出现过一次）。

    判据：把标点去掉后整句话本身就是提示词的一个片段 —— 自己组织的话不会正好是提示的子串。
    """
    if not hint or not text:
        return False
    norm = lambda x: _re.sub(r"[\s，,。.；;：:！!？?、（）()「」\"'*]", "", _re.sub(r"[（(].*?[)）]", "", x))
    t, h = norm(text), norm(hint)
    return len(t) >= 8 and t in h


def check_persona(text: str) -> tuple[bool, str]:
    """教练不在游戏里：第一人称动作（"等我扶他"）= 把自己当队友，打回重写（P1v10 28:43 审片标注）。"""
    m = _PERSONA_BAD.search(text)
    return (False, f"教练把自己当队友了：{m.group(0)}") if m else (True, "")


def strip_direction(text: str) -> str:
    """兜底：把不允许的方向短语删掉（"往右手边他那边靠"→"往他那边靠"，"往西北方向跑进圈"→"跑进圈"，"在左边挨打"→"挨打"）。"""
    t = _BAD_TARGET.sub(lambda m: m.group(0)[len(m.group(1)):].lstrip("的"), text)
    t = _re.sub(r"(往|向|朝|在|去|到)" + _DIR + r"(方向|那边|边)?", "", t)
    t = _re.sub(_DIR + r"(方向)?(的)?(开阔地|空地|草地|路)", r"\3", t)
    return t.replace("，，", "，").replace("！，", "！").strip("，")
