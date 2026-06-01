"""统一加载 4 个 Webb 任务的人类(和原文 GPT-3)聚合基线,
分析层把它们当作 model 行追加到 by-task / by-condition 长表中,
柱状图能自然把"Human"/"GPT-3 (paper)"画成 bar(而非 dashed line)。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.config import project_root


REPO = project_root() / "data" / "repo_original"


# ============ Digit Matrices ============

def digit_matrix_human_by_class() -> dict[str, float]:
    """4 problem class 平均人类准确率(来自 probcat_gen_acc_behavior.npz)。"""
    p = REPO / "digit_mat" / "exp1_behavioral_data" / "probcat_gen_acc_behavior.npz"
    arr = np.load(p, allow_pickle=True)["acc"]
    return {"one_rule": float(arr[0]), "two_rule": float(arr[1]),
            "three_rule": float(arr[2]), "logic": float(arr[3])}


def digit_matrix_gpt3_by_class() -> dict[str, float]:
    """原文 GPT-3 (text-davinci-003) generative 在 4 个 class 平均的准确率。
    用每子类型 acc → 按 problem_class 分组平均。"""
    p = REPO / "digit_mat" / "gpt_matprob_results.npz"
    blob = np.load(p, allow_pickle=True)
    gen = blob["all_gen_correct_pred"].item()
    from ..tasks.digit_matrix import problem_class
    by_cls = {"one_rule": [], "two_rule": [], "three_rule": [], "logic": []}
    for st, vec in gen.items():
        arr = np.array(vec).astype(bool)
        if len(arr) == 0:
            continue
        cls = problem_class(st)
        if cls in by_cls:
            by_cls[cls].extend(arr.tolist())
    return {k: float(np.mean(v)) if v else float("nan") for k, v in by_cls.items()}


# ============ Letter String ============

def letter_string_human_by_gen() -> dict[int, float]:
    """4 个 generalization level 上的人类平均准确率(all_gen_acc.npz 的 4 个总值)。"""
    p = REPO / "letter_string" / "behavioral_results" / "all_gen_acc.npz"
    arr = np.load(p, allow_pickle=True)["all_acc"]
    return {i: float(arr[i]) for i in range(len(arr))}


# ============ Verbal — UCLA VAT ============

def ucla_vat_human_by_relation() -> dict[str, float]:
    """57 subj × 4 relation cols (单位是百分制,除 100)。"""
    p = REPO / "UCLA_VAT" / "UCLA_VAT_ind_subj_data.xlsx"
    df = pd.read_excel(p, sheet_name="ind_subj").dropna()
    means = df.mean() / 100.0
    return {k: float(v) for k, v in means.items()}


def ucla_vat_human_total() -> float:
    return float(np.mean(list(ucla_vat_human_by_relation().values())))


def ucla_vat_gpt3_total() -> float:
    """原文报告 GPT-3 (text-davinci-003) 在 UCLA VAT 整体 ~0.80。
    优先从仓库 UCLA_VAT_results.npz 抽,失败则用论文报告值。"""
    p = REPO / "UCLA_VAT" / "UCLA_VAT_results.npz"
    if p.exists():
        blob = np.load(p, allow_pickle=True)
        for k in blob.files:
            arr = blob[k]
            if arr.dtype.kind in "biu" and arr.ndim == 1:
                return float(np.mean(arr))
    return 0.80   # paper 报告值兜底


# ============ Verbal — Sternberg / Kmiecik (no per-item human, paper 聚合) ============

def sternberg_paper_baselines() -> dict[str, float]:
    """Webb 2023 paper Fig.7 报告:GPT-3 ~0.78, Human ~0.90 在 Sternberg & Nigro。
    (这些是 paper 报告的均值,非 per-item)"""
    return {"gpt3": 0.78, "human": 0.90}


def kmiecik_paper_baselines() -> dict[str, float]:
    """Webb 2023 paper 中 Jones et al. 2022 / Kmiecik 数据上的近似均值。
    paper 报告 GPT-3 在 Jones 上 ~0.78,人类 ~0.80(EEG 行为)。
    """
    return {"gpt3": 0.78, "human": 0.80}


# ============ Story Analogies ============

def story_baselines_by_condition() -> pd.DataFrame:
    """从 human_vs_gpt3_data.csv 算 human / GPT-3 在 analogy vs similarity 两条件的准确率。
    返回 long 表:agent, condition, acc"""
    p = REPO / "story_analogies" / "human_vs_gpt3_data.csv"
    df = pd.read_csv(p)
    g = (df.groupby(["human_vs_gpt", "analogy_vs_similarity"])
         ["correct_pred"].mean().reset_index())
    g["agent"] = g["human_vs_gpt"].map({0: "Human (paper)", 1: "GPT-3 (paper)"})
    g["condition"] = g["analogy_vs_similarity"].map({0: "similarity", 1: "analogy"})
    return g[["agent", "condition", "correct_pred"]].rename(columns={"correct_pred": "acc"})


def gpt4_story_baselines_by_condition() -> pd.DataFrame:
    """gpt4_data.csv 中 GPT-4 在 analogy/similarity 两条件的准确率。"""
    p = REPO / "story_analogies" / "gpt4_data.csv"
    if not p.exists():
        return pd.DataFrame(columns=["agent", "condition", "acc"])
    df = pd.read_csv(p)
    g = df.groupby("analogy_vs_similarity")["correct_pred"].mean().reset_index()
    g["agent"] = "GPT-4 (Webb follow-up)"
    g["condition"] = g["analogy_vs_similarity"].map({0: "similarity", 1: "analogy"})
    return g[["agent", "condition", "correct_pred"]].rename(columns={"correct_pred": "acc"})
