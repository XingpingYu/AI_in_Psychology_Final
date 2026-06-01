"""从 raw jsonl 里挖代表性 case,组装成 case_studies.md 与 case_studies.json。

按以下角度选 case:
  A. 'completion-style bias' — gpt-3.5-instruct 答整行而非单格
  B. '人难题模型也难' — letter_string gen-3 上的成败对照
  C. 'reasoning 救活 chat' — 同题:chat 错而 reasoner 对,展示 thinking 关键步骤
  D. '模型间分裂' — 一道 digit matrix 上,top模型对 / weak 模型错,体现规模差异
  E. '超人题' — 人类 logic 类 acc 39% 而 top 模型 ≥85%,挑一道 logic 题展示模型如何"看穿"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import jsonl_read
from src.utils.config import load_experiment


def load_records(raw_dir: Path, task: str) -> dict[str, dict[str, dict]]:
    """返回 records[item_id][model_id] = record dict。"""
    by_item: dict[str, dict[str, dict]] = {}
    for model_dir in raw_dir.iterdir():
        if not model_dir.is_dir():
            continue
        jf = model_dir / f"{task}.jsonl"
        if not jf.exists():
            continue
        for r in jsonl_read(jf):
            it = r.get("item_id")
            if it is None:
                continue
            by_item.setdefault(it, {})[model_dir.name] = r
    return by_item


def fmt_prompt(p: str, max_len: int = 600) -> str:
    return (p[:max_len] + "  …(truncated)") if len(p) > max_len else p


def case_A_completion_bias(by_item):
    """A. gpt-3.5-instruct 在 one_rule digit_matrix 上的整行复读"""
    # 选一道答案是单格的题
    target = None
    for it, recs in by_item.items():
        r35 = recs.get("gpt-3.5-turbo-instruct")
        if r35 is None or r35.get("subtype") != "row_constant":
            continue
        # gpt-3.5 错,gpt-5 / o3-mini 对
        if not r35.get("correct", False) and \
           recs.get("gpt-5", {}).get("correct") and \
           recs.get("o3-mini", {}).get("correct"):
            target = (it, recs)
            break
    return target


def case_B_letter_string_hard(by_item):
    """B. letter_string gen-3 上多数模型都错的题"""
    target = None
    for it, recs in by_item.items():
        if not it.startswith("group_letter2num"):  # 一种 gen 多的子类型
            pass
        # 选有 ≥10 个模型记录、其中 ≤3 个对的题
        if len(recs) < 10:
            continue
        n_correct = sum(1 for r in recs.values() if r.get("correct"))
        if n_correct == 0 or n_correct > 3:
            continue
        # 至少有一个 reasoner 对
        if not (recs.get("o4-mini", {}).get("correct") or
                recs.get("gpt-5", {}).get("correct") or
                recs.get("deepseek-reasoner", {}).get("correct")):
            continue
        target = (it, recs)
        break
    return target


def case_C_reasoner_saves(by_item):
    """C. 同题:deepseek-chat 错而 deepseek-reasoner 对 → 展示 thinking"""
    for it, recs in by_item.items():
        ch = recs.get("deepseek-chat")
        rs = recs.get("deepseek-reasoner")
        if not (ch and rs):
            continue
        if ch.get("correct"):
            continue
        if not rs.get("correct"):
            continue
        # 优先选 logic 或 three_rule
        sub = ch.get("subtype", "")
        if not (sub.endswith("_set_union") or sub.startswith("AND") or
                sub.startswith("XOR") or sub.startswith("three_rule")):
            continue
        # 选有 reasoning_text 的
        if not rs.get("reasoning_text"):
            continue
        return (it, recs)
    return None


def case_D_model_split(by_item):
    """D. 模型间分裂:gpt-4.1-nano / gpt-4o-mini 错,gpt-4.1 / gpt-5-mini / o3-mini 对"""
    # 挑一道 two_rule 题
    for it, recs in by_item.items():
        if not any(it.startswith(s) for s in ("two_rule_comb", "three_rule_comb")):
            continue
        weak = ["gpt-4.1-nano", "gpt-4o-mini"]
        strong = ["gpt-4.1", "gpt-5-mini", "o3-mini"]
        if all(recs.get(m, {}).get("correct") is False for m in weak) and \
           all(recs.get(m, {}).get("correct") for m in strong):
            return (it, recs)
    return None


def case_E_logic_superhuman(by_item):
    """E. logic 题 — human 39%,top 模型 ≥85%。挑一道 AND/XOR/c2_set_union 类型,
    展示 chat-level 模型(deepseek-chat / gpt-4o-mini)经常错,而 top 都对"""
    for it, recs in by_item.items():
        sub = (recs.get(next(iter(recs)), {}) or {}).get("subtype", "")
        if not (sub in ("AND", "XOR", "c2_set_union", "c3_set_union")):
            continue
        chat_models = ["deepseek-chat", "gpt-4o-mini", "gpt-4.1-mini"]
        top_models = ["o3-mini", "o4-mini", "gpt-5", "gpt-5-mini"]
        n_chat_wrong = sum(1 for m in chat_models if recs.get(m, {}).get("correct") is False)
        n_top_right = sum(1 for m in top_models if recs.get(m, {}).get("correct"))
        if n_chat_wrong >= 2 and n_top_right >= 3:
            return (it, recs)
    return None


_LS_ANSWER_CACHE: dict | None = None


def _ls_answer_lookup(item_id: str):
    """letter_string 的 correct 不在 jsonl 里,从 task items 反查 tgt_B。"""
    global _LS_ANSWER_CACHE
    if _LS_ANSWER_CACHE is None:
        from src.tasks import letter_string as t_ls
        _LS_ANSWER_CACHE = {it.item_id: list(it.payload["tgt_B"])
                            for it in t_ls.load_items(n_per_subtype=None)}
    return _LS_ANSWER_CACHE.get(item_id)


def render_case(name: str, head: str, item_id: str, recs: dict[str, dict],
                models_to_show: list[str], show_reasoning_for: list[str] = ()) -> str:
    md = [f"### {name}: {head}", "", f"`item_id = {item_id}`",
          f"**Subtype**: `{recs[next(iter(recs))].get('subtype')}`", ""]
    # 给一份 prompt 与正确答案
    sample = next(iter(recs.values()))
    md.append("**Prompt(模型实际看到的)**:")
    md.append("```")
    md.append(fmt_prompt(sample.get("prompt", "")))
    md.append("```")
    md.append("")
    correct_ans = sample.get("correct_answer")
    if correct_ans is None and sample.get("task") == "letter_string":
        correct_ans = _ls_answer_lookup(item_id)
    md.append(f"**正确答案**:`{correct_ans}`")
    md.append("")
    md.append("**各模型响应**:")
    md.append("")
    md.append("| Model | Correct? | Parsed | Response (head) |")
    md.append("|---|:-:|---|---|")
    for m in models_to_show:
        r = recs.get(m)
        if not r:
            md.append(f"| {m} | — | — | (未跑) |")
            continue
        ok = "✅" if r.get("correct") else "❌"
        parsed = r.get("parsed")
        rt = (r.get("response_text") or "")[:140].replace("\n", " ⏎ ")
        if not rt and r.get("reasoning_text"):
            rt = "[thinking 截断,无 response_text]"
        md.append(f"| {m} | {ok} | `{parsed}` | {rt} |")
    md.append("")
    # 可选:展示某模型的 thinking 片段
    for m in show_reasoning_for:
        r = recs.get(m)
        if not r:
            continue
        rt = r.get("reasoning_text") or ""
        if not rt:
            continue
        # 取前 250 字 + 后 250 字
        snippet = rt[:400] if len(rt) <= 800 else rt[:300] + "\n…(省略中间 thinking)…\n" + rt[-250:]
        md.append(f"**{m} 的 thinking 片段**:")
        md.append("```")
        md.append(snippet)
        md.append("```")
        md.append("")
    return "\n".join(md)


# ---------- 主入口 ----------
MODELS_FOR_TABLE = [
    "gpt-3.5-turbo-instruct", "gpt-4o-mini", "gpt-4o",
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    "gpt-5-nano", "gpt-5-mini", "gpt-5",
    "o3-mini", "o4-mini",
    "deepseek-chat", "deepseek-reasoner",
]


def main():
    exp = load_experiment()
    raw_dir = ROOT / exp["paths"]["raw_results_dir"]

    dm = load_records(raw_dir, "digit_matrix")
    ls = load_records(raw_dir, "letter_string")

    out = ["# Case Studies (从 8,033 条 raw 响应里挑选)", "",
           "本文档由 `scripts/case_studies.py` 自动生成,"
           "展示模型在不同任务上的代表性成败 case,"
           "用于补充报告 §11。", ""]

    # 案例 A
    A = case_A_completion_bias(dm)
    if A:
        it, recs = A
        out.append(render_case(
            name="Case A",
            head="completion-style bias — `gpt-3.5-turbo-instruct` 把单格补成整行",
            item_id=it, recs=recs,
            models_to_show=["gpt-3.5-turbo-instruct", "gpt-4o-mini", "gpt-4.1",
                            "gpt-5", "o3-mini", "deepseek-chat", "deepseek-reasoner"],
        ))
        out.append("**[解读]** 单格答案在 chat 模型上都没问题,而 `gpt-3.5-turbo-instruct` 把 `[6] [6] [?]` 补成 `[6 6 6]` 整行 — completion 模型对格式化模板的自动补全先验击败了我们的 `reply only the missing cell` 指令。要让它正确答题需要走原文 logprob 协议,而非 chat-style prompt。")
        out.append("")

    # 案例 B
    B = case_B_letter_string_hard(ls)
    if B:
        it, recs = B
        out.append(render_case(
            name="Case B",
            head="Letter String gen-3:多数模型失败的真正难题",
            item_id=it, recs=recs,
            models_to_show=MODELS_FOR_TABLE,
            show_reasoning_for=["deepseek-reasoner"],
        ))
        out.append("**[解读]** 这是一道 `fix alphabetic order` — src_A 的最后一个字符 `j` 不连续,要改成 `i` 让序列连续;src_B 的首字符 `s` 不连续 (与 `w` 不衔接),要改成 `v`,得到 `[v w x y z]`。**13 个模型里只有 gpt-5 答对**;其他大多数模型采用了'局部修改最后一个'的浅启发式 (`[s w x y a]`/`[s w x y y]`/`[s w x y z]`),没有意识到出错的元素其实在序列开头。这是 emergent analogy 'far transfer' 上的真实瓶颈。")
        out.append("")

    # 案例 C
    C = case_C_reasoner_saves(dm)
    if C:
        it, recs = C
        out.append(render_case(
            name="Case C",
            head="Reasoning 'rescue':`deepseek-chat` 错,`deepseek-reasoner` 对",
            item_id=it, recs=recs,
            models_to_show=["deepseek-chat", "deepseek-reasoner", "gpt-4o", "o4-mini"],
            show_reasoning_for=["deepseek-reasoner"],
        ))
        out.append("**[解读]** thinking 步骤里能看到 reasoner 显式枚举/验证规则;chat 在没有这种结构时往往凭直觉给一个 plausible 但错的答案。这是 reasoning vs chat 提升的微观证据。")
        out.append("")

    # 案例 D
    D = case_D_model_split(dm)
    if D:
        it, recs = D
        out.append(render_case(
            name="Case D",
            head="模型规模分裂 — small 错,大模型对",
            item_id=it, recs=recs,
            models_to_show=["gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini",
                            "gpt-4.1", "gpt-5-mini", "o3-mini"],
        ))
        out.append("**[解读]** 这类 case 是 Ext2 规模效应曲线的典型驱动 — 题目需要 model 整合两条规则,nano 档没能力,full 档稳过。")
        out.append("")

    # 案例 E
    E = case_E_logic_superhuman(dm)
    if E:
        it, recs = E
        out.append(render_case(
            name="Case E",
            head="Logic 类'超人'case — human 39%,top 模型 ≥85%",
            item_id=it, recs=recs,
            models_to_show=["deepseek-chat", "gpt-4o-mini", "gpt-4.1-mini",
                            "deepseek-reasoner", "gpt-5", "o3-mini", "o4-mini"],
        ))
        out.append("**[解读]** logic 类(AND/XOR/set-union 等)是 human 最弱 (39%) 的子类,但 frontier 模型已经在这一类上彻底碾压 — 这是 emergent reasoning 在 4 年内最大的拐点。")
        out.append("")

    text = "\n".join(out)
    out_path = ROOT / "reports" / "case_studies.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path}  ({len(text)} chars)")


if __name__ == "__main__":
    main()
