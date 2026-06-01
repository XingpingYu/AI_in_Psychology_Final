"""三个拓展方向的统一计算函数。

Ext1: reasoning vs chat — 按 family 配对
Ext2: scale effect — 同 family 不同 size
Ext3: human-similarity — 在 4 个 problem class 上 model vs human acc 的 r
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def ext1_reasoning_vs_chat(df_ok: pd.DataFrame, model_meta: pd.DataFrame,
                            pairs: list[tuple[str, str, str]] | None = None
                            ) -> pd.DataFrame:
    """
    返回 cols: pair_label, chat_id, reasoner_id, task, chat_acc, reasoner_acc, delta

    pairs 形如 [(pair_label, chat_id, reasoner_id), ...]。如果不传,默认使用
    人工定义的跨/同家族对:
        - DeepSeek: chat vs reasoner (同家族)
        - OpenAI mid: gpt-4o-mini vs o3-mini (同等价位,显式 reasoning)
        - OpenAI mini-new: gpt-4.1-mini vs o4-mini
        - GPT-5 vs o4-mini(都是 frontier,但 gpt-5 是 chat 训练 + 隐式 reasoning,
          o4-mini 是显式 reasoning 训练,粗糙对比)
    """
    if pairs is None:
        pairs = [
            ("DeepSeek (same-family)", "deepseek-chat", "deepseek-reasoner"),
            ("OpenAI mid (gpt-4o-mini vs o3-mini)", "gpt-4o-mini", "o3-mini"),
            ("OpenAI new (gpt-4.1-mini vs o4-mini)", "gpt-4.1-mini", "o4-mini"),
        ]
    out = []
    avail = set(df_ok["model"].unique())
    for label, chat_id, rs_id in pairs:
        if not {chat_id, rs_id}.issubset(avail):
            continue
        for task in df_ok["task"].unique():
            d = df_ok[df_ok["task"] == task]
            if not {chat_id, rs_id}.issubset(set(d["model"].unique())):
                continue
            ca = d.loc[d["model"] == chat_id, "correct"].mean()
            ra = d.loc[d["model"] == rs_id, "correct"].mean()
            out.append({"pair_label": label, "chat_id": chat_id,
                        "reasoner_id": rs_id, "task": task,
                        "chat_acc": ca, "reasoner_acc": ra,
                        "delta": ra - ca})
    return pd.DataFrame(out)


def ext2_scale_effect(df_ok: pd.DataFrame, model_meta: pd.DataFrame) -> pd.DataFrame:
    """
    同 family 内,把 size 作为顺序变量(对闭源 OpenAI 用 family 内的 size_rank 替代真实参数量),
    每 (family, task) 算 model_id 对应的 acc。
    返回 long 表: family, model_id, size_rank, task, acc
    """
    meta = model_meta.copy()
    out = []
    # 对每个 family,如果显式 size_b 缺失,用模型名末尾约定 nano/mini/full 的 rank 0/1/2
    SIZE_HINT = {"nano": 0, "mini": 1, "full": 2, "": 2}
    def rank_for(row):
        if pd.notna(row.get("size_b")):
            try:
                return float(row["size_b"])
            except Exception:
                pass
        nm = row["id"]
        if "nano" in nm: return 0
        if "mini" in nm: return 1
        return 2
    meta["size_rank"] = meta.apply(rank_for, axis=1)
    for fam, gf in meta.groupby("family"):
        if len(gf) < 2:
            continue
        # 同家族里至少 2 个 size
        gf = gf.sort_values("size_rank")
        for _, row in gf.iterrows():
            mid = row["id"]
            for task in df_ok["task"].unique():
                d = df_ok[(df_ok["task"] == task) & (df_ok["model"] == mid)]
                if len(d) == 0:
                    continue
                out.append({"family": fam, "model_id": mid,
                            "size_rank": row["size_rank"],
                            "is_reasoning": bool(row.get("reasoning", False)),
                            "task": task,
                            "n": len(d), "acc": d["correct"].mean()})
    return pd.DataFrame(out)


def ext3_human_similarity(df_ok: pd.DataFrame, human_class_acc: dict) -> pd.DataFrame:
    """对每个模型计算 digit_matrix 4-class 准确率与人类 4-class 准确率的 r。
    human_class_acc: {one_rule: float, two_rule: float, three_rule: float, logic: float}
    """
    dm = df_ok[df_ok["task"] == "digit_matrix"].copy()
    # problem_class 需在 attach_problem_class 后存在
    if "problem_class" not in dm.columns:
        from .loader import attach_problem_class
        dm = attach_problem_class(dm.assign(task="digit_matrix"))
    classes = ["one_rule", "two_rule", "three_rule", "logic"]
    out = []
    for m in dm["model"].unique():
        sub = dm[dm["model"] == m]
        accs = []
        for c in classes:
            v = sub[sub["problem_class"] == c]["correct"].mean()
            if pd.notna(v):
                accs.append((c, v))
        if len(accs) < 3:
            r = np.nan
        else:
            xs = np.array([human_class_acc[c] for c, _ in accs])
            ys = np.array([v for _, v in accs])
            r = float(np.corrcoef(xs, ys)[0, 1])
        out.append({"model": m, "r_vs_human_4class": r,
                    "n_classes_compared": len(accs)})
    return pd.DataFrame(out)


def load_human_class_acc() -> dict:
    """从原仓库 digit_mat/exp1_behavioral_data/probcat_gen_acc_behavior.npz 读取人类
    4 个 problem class 的总准确率。
    返回顺序约定:[one_rule, two_rule, three_rule, logic]
    """
    from ..utils.config import project_root
    p = project_root() / "data" / "repo_original" / "digit_mat" / "exp1_behavioral_data" / "probcat_gen_acc_behavior.npz"
    blob = np.load(p, allow_pickle=True)
    acc = blob["acc"]
    # 原仓库的 R 脚本里约定的顺序就是 [one, two, three, logic]
    return {
        "one_rule": float(acc[0]),
        "two_rule": float(acc[1]),
        "three_rule": float(acc[2]),
        "logic": float(acc[3]),
    }
