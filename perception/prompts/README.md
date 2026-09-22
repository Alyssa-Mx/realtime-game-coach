# VLM 预标注调用约定（v1）

这是**建标注数据阶段**（还没有自己的识别模型时）用的提示词约定：一律本地模型推理。
- VLM：**Qwen3-VL-8B-Instruct**，vLLM 离线批推，温度 0，只输出 JSON。升级备选：Qwen3-VL-30B-A3B-Instruct（MoE 激活 3B）。
- OCR 选型原则：**建真值优先判别式**（CTC 逐字符置信度、白名单可在解码层限死、
  读不出→低分自动进 unclear）；生成式 OCR（DeepSeek-OCR 等）错时会产出通顺的幻觉且无
  字符级置信度，混入真值集无法察觉 —— 不当主力，只当被测候选。
  · 数字字段（弹药/倒计时/剩余/距离/速度）：PaddleOCR + 白名单 ↔ 逐字模板匹配，30 帧实测谁准用谁
  · 中文文本行（横幅/击杀记录/队友消息）：**双读** —— Qwen3-VL（prompt 照抄）+ PaddleOCR
    各跑一遍，逐字一致 → 直接采信；不一致 → 进人工队列。两个模型错法不同，撞车即真值。
  · DeepSeek-OCR（盘上 6.3G）：第三候选，仅在双读分歧率过高的字段上参评
  · PaddleOCR 装不上（个别发行版偶发）→ RapidOCR（同模型 ONNX 版，纯 pip）
- 验收门：每字段 30 帧人工对照一致率 ≥90%，不达标先改 prompt 再换模型。

## 统一规则（每个 prompt 文件都遵守）
1. **裁剪后喂图，不喂整帧**：按 layout.json 的 bbox 裁剪，四周各留 8px，
   最近邻放大 3-4 倍再发送（小图标在整帧里只有十几像素，压缩后不可读）。
2. system 固定为：「你是标注员。只输出 JSON，不输出任何其他文字。
   只记录画面里直接可见的事实。看不清或不存在的用 null / unknown / 空数组，禁止猜测。」
3. 校验回包：JSON 可解析 + 枚举合法。失败带错误信息重试 1 次，再失败 → 该字段进 unclear。
4. 一次调用只管一个区域（好校验、好归因）。别把六个区域塞一个 prompt。
5. 每批先抽 30 帧和人工对答案，一致率 <90% 的字段回来改 prompt，别硬跑全量。

## 字段与 prompt 对应
| 字段 | prompt 文件 | 裁剪区域（layout.json 键名） |
|---|---|---|
| minimap.* | minimap.txt | 小地图 |
| banner | banner.txt | 主视角击杀横幅 |
| killfeed | killfeed.txt | 他人击杀记录 |
| team_msgs | team_msgs.txt | 队友信息条 |
| team | team_panel.txt | 队友状态栏 |
| weapon_main/alt | weapons.txt | 武器栏 |

hp_ratio / energy_ratio / 横杠 / 物资数字 → 不走 VLM，确定性管线。
