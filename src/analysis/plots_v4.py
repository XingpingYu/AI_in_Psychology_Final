"""v4 画图模块。
风格目标 — 接近 Webb 2023 原论文的 Nature Hum Behav 风格:
  - Human / GPT-3 / GPT-4 paper 作为"参照 model"画成 bar 与新模型并列
  - 每个模型 family 用同色调梯度:小→浅,大→深;reference 用 灰阶
  - 4 任务横排或竖排,Bar 间距宽松,legend 放图外右侧分两列
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42


def _ramp(cmap_name: str, n: int, lo: float = 0.30, hi: float = 0.92) -> list[str]:
    """从 matplotlib sequential cmap 等距取 n 个色,避开极端两端。"""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap(cmap_name)
    if n == 1:
        positions = [(lo + hi) / 2]
    else:
        positions = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    return [mcolors.to_hex(cmap(p)) for p in positions]


# ---- v4.2 调色板 — 手挑 + 单 family 单色调梯度 ----
# 设计原则:
#  - 参照 baseline 用灰阶,Human 最深;
#  - 7 个不同模型组用 7 个**清晰可辨**的色调,family 内部 small→full = 浅→深;
#  - 颜色都 desaturated(像 Nature 风格),避免"玩具"感;
#  - 相邻 family 在主图柱状里不会撞色(gpt-4o teal ≠ gpt-4.1 green ≠ o-reasoning navy)。

PALETTE = {
    # references — 灰阶
    "Human (paper)":            "#1f1f1f",     # 近黑
    "GPT-3 (paper)":            "#7e7e7e",     # 中灰
    "GPT-4 (Webb follow-up)":   "#c2c2c2",     # 浅灰

    # gpt-3.5-turbo-instruct — 沙棕 (单色,与 gpt-5 暖色族明显分开)
    "gpt-3.5-turbo-instruct":   "#b89066",

    # gpt-4o (2 sizes) — teal,既不同绿也不同蓝
    "gpt-4o-mini":              "#7eb2ad",     # 浅 teal
    "gpt-4o":                   "#2a5f5a",     # 深 teal

    # gpt-4.1 (3 sizes) — 绿(yellow-green → forest)
    "gpt-4.1-nano":             "#b9d68f",     # 浅黄绿
    "gpt-4.1-mini":             "#6fa14a",     # 中绿
    "gpt-4.1":                  "#2d6324",     # 深森林绿

    # gpt-5 (3 sizes) — 暖红
    "gpt-5-nano":               "#f4a18d",     # 浅珊瑚
    "gpt-5-mini":               "#d96444",     # 中橙红
    "gpt-5":                    "#8a1d1c",     # 深胭脂红

    # o-reasoning (2 sizes) — 蓝(steel → navy)
    "o3-mini":                  "#92b6d5",     # 浅钢蓝
    "o4-mini":                  "#1e4a6e",     # 深海军蓝

    # deepseek (2 sizes) — 紫
    "deepseek-chat":            "#b69ac8",     # 浅薰衣草
    "deepseek-reasoner":        "#5d2b87",     # 深紫
}


def _short(label: str) -> str:
    return (label.replace("deepseek-", "ds-")
                  .replace("gpt-3.5-turbo-instruct", "gpt-3.5-inst")
                  .replace(" (paper)", "*")
                  .replace(" (Webb follow-up)", "*"))


# 按 family 排序模型(Webb 风格 — small→large,human/GPT-3 在两端)
MODEL_ORDER = [
    "Human (paper)",
    "GPT-3 (paper)",
    "GPT-4 (Webb follow-up)",
    "gpt-3.5-turbo-instruct",
    "gpt-4o-mini", "gpt-4o",
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    "gpt-5-nano", "gpt-5-mini", "gpt-5",
    "o3-mini", "o4-mini",
    "deepseek-chat", "deepseek-reasoner",
]


def _ordered(df: pd.DataFrame, col: str = "model") -> list[str]:
    """返回 df 中模型按 MODEL_ORDER 排序的列表。"""
    have = set(df[col].unique())
    return [m for m in MODEL_ORDER if m in have]


# ---------------------------------------------------------------------------

def plot_grouped_acc(df: pd.DataFrame, x: str, y: str, hue: str,
                     out: Path, *,
                     x_order: list[str] | None = None,
                     hue_order: list[str] | None = None,
                     title: str = "", ylim=(0, 1.05),
                     ylabel: str = "Accuracy",
                     chance: float | None = None,
                     ax_size=(15, 5.5),
                     legend_ncol: int = 1) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=ax_size)
    hue_order = hue_order or sorted(df[hue].unique())
    palette = [PALETTE.get(m, "#999999") for m in hue_order]
    sns.barplot(data=df, x=x, y=y, hue=hue,
                order=x_order, hue_order=hue_order,
                palette=palette, edgecolor="white", linewidth=0.5,
                ax=ax, errorbar=None, saturation=0.80)
    if chance is not None:
        ax.axhline(chance, color="#666666", linestyle="--", linewidth=0.9, alpha=0.7,
                   zorder=0, label=f"chance = {chance:.0%}")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=12, pad=10, loc="left")
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    # legend 右外侧分列;短名显示
    handles, labels = ax.get_legend_handles_labels()
    short_labels = [_short(l) for l in labels]
    leg = ax.legend(handles, short_labels,
                    loc="center left", bbox_to_anchor=(1.005, 0.5),
                    fontsize=8.5, frameon=False, ncol=legend_ncol,
                    handletextpad=0.5, columnspacing=0.8, labelspacing=0.5,
                    borderaxespad=0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_overall_with_humans(by_task: pd.DataFrame, humans: dict[str, float],
                              gpt3: dict[str, float], out: Path) -> None:
    """4 任务总览,human 与 GPT-3 作为额外 bar 并列。"""
    extras = []
    for task, acc in humans.items():
        extras.append({"model": "Human (paper)", "task": task, "acc": acc})
    for task, acc in gpt3.items():
        if not np.isnan(acc):
            extras.append({"model": "GPT-3 (paper)", "task": task, "acc": acc})
    full = pd.concat([by_task, pd.DataFrame(extras)], ignore_index=True)
    hue_order = _ordered(full)
    # 4 tasks x 15 bars: 用更宽,legend 双列
    plot_grouped_acc(full, x="task", y="acc", hue="model", out=out,
                     hue_order=hue_order,
                     title="Overall accuracy across Webb 2023's 4 tasks",
                     ax_size=(14, 5.0), legend_ncol=1)


def plot_digit_matrix_4class(df_long: pd.DataFrame,
                              human: dict[str, float],
                              gpt3: dict[str, float],
                              out: Path) -> None:
    """Fig.3 风格:X = problem_class,bar = model + human + GPT-3。"""
    classes = ["one_rule", "two_rule", "three_rule", "logic"]
    extras = []
    for c in classes:
        extras.append({"model": "Human (paper)", "problem_class": c, "acc": human[c]})
        extras.append({"model": "GPT-3 (paper)", "problem_class": c, "acc": gpt3[c]})
    full = pd.concat([df_long, pd.DataFrame(extras)], ignore_index=True)
    hue_order = _ordered(full)
    plot_grouped_acc(full, x="problem_class", y="acc", hue="model", out=out,
                     x_order=classes, hue_order=hue_order,
                     title="Digit Matrices: accuracy by problem class",
                     ax_size=(14, 5.0))


def plot_letter_string_by_gen(df_long: pd.DataFrame,
                               human: dict[int, float],
                               out: Path) -> None:
    """letter_string: X = gen_level (0/1/2/3), bar = model + human。"""
    levels = sorted(human.keys())
    extras = []
    for g in levels:
        extras.append({"model": "Human (paper)", "gen_level": g, "acc": human[g]})
    full = pd.concat([df_long, pd.DataFrame(extras)], ignore_index=True)
    full["gen_level_str"] = "gen=" + full["gen_level"].astype(str)
    hue_order = _ordered(full)
    plot_grouped_acc(full, x="gen_level_str", y="acc", hue="model", out=out,
                     x_order=[f"gen={g}" for g in levels], hue_order=hue_order,
                     title="Letter String Analogies: accuracy by generalization level",
                     ax_size=(14, 5.0))


def plot_verbal_panel(by_task_long: pd.DataFrame,
                       humans: dict[str, float],
                       gpt3s: dict[str, float],
                       out: Path) -> None:
    """X = verbal_dataset (ucla_vat / sternberg / kmiecik),bar = model + Human + GPT-3。"""
    extras = []
    for d, v in humans.items():
        extras.append({"model": "Human (paper)", "task": d, "acc": v})
    for d, v in gpt3s.items():
        extras.append({"model": "GPT-3 (paper)", "task": d, "acc": v})
    full = pd.concat([by_task_long, pd.DataFrame(extras)], ignore_index=True)
    # 把 task 名美化
    nice = {"verbal_ucla_vat": "UCLA VAT (80)",
            "verbal_sternberg": "Sternberg 1980 (197)",
            "verbal_kmiecik":   "Kmiecik / Jones 2022 (160)"}
    full["task"] = full["task"].map(nice).fillna(full["task"])
    hue_order = _ordered(full)
    plot_grouped_acc(full, x="task", y="acc", hue="model", out=out,
                     x_order=list(nice.values()), hue_order=hue_order,
                     title="Four-term Verbal Analogies (chance = 50%)",
                     chance=0.5, ax_size=(13.5, 5.0))


def plot_story_commit_vs_acc(by_cond_full: pd.DataFrame, out: Path) -> None:
    """在 strict-acc 与 commit_rate 两个维度上的散点图,
    凸显"承诺度低的模型 strict-acc 也低"这个 RLHF-hedging 模式。"""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.5, 7))
    g = by_cond_full.groupby("model").agg(
        strict=("acc", "mean"), commit=("commit_rate", "mean"),
        lenient=("lenient_acc", "mean")).reset_index()
    # 防注释重叠 - 偏移规则
    seen = []
    for _, r in g.iterrows():
        c = PALETTE.get(r["model"], "#999999")
        ax.scatter(r["commit"], r["strict"], s=180, color=c, edgecolor="black",
                   linewidth=0.6, zorder=3, alpha=0.9)
        # 简单避让:每个邻近点偏不同方向
        dx, dy = 8, 4
        for sx, sy in seen:
            if abs(sx - r["commit"]) < 0.04 and abs(sy - r["strict"]) < 0.04:
                dy -= 12
        ax.annotate(_short(r["model"]), (r["commit"], r["strict"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    color="#333333")
        seen.append((r["commit"], r["strict"]))
    ax.plot([0, 1], [0, 1], "--", color="#888888", alpha=0.6, linewidth=1.2,
            label="y = commit rate (upper bound)")
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Commit rate  (1 − 'both equally' rate)", fontsize=11)
    ax.set_ylabel("Strict accuracy  (both = wrong)", fontsize=11)
    ax.set_title("Story Analogies — hedging vs accuracy", fontsize=12, loc="left", pad=10)
    # 副标题作为子文本
    fig.text(0.5, 0.91,
             "top-right = commits and is right · bottom-left = hedges and is also wrong",
             ha="center", fontsize=9, style="italic", color="#555555")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


# -------------------------- Ext 1/2/3 figures --------------------------

def plot_ext1_reasoning_vs_chat(ext1: pd.DataFrame, out: Path) -> None:
    """Ext 1: 3 pair × 6 task 的 Δ(reasoner − chat)条形图。"""
    setup_style()
    fig, ax = plt.subplots(figsize=(13, 5))
    pretty = {"digit_matrix": "Digit\nMatrices",
              "letter_string": "Letter\nString",
              "verbal_ucla_vat": "Verbal\nUCLA VAT",
              "verbal_sternberg": "Verbal\nSternberg",
              "verbal_kmiecik": "Verbal\nKmiecik",
              "story_analogy": "Story\nAnalogies"}
    order = [t for t in ["digit_matrix", "letter_string",
                          "verbal_ucla_vat", "verbal_sternberg", "verbal_kmiecik",
                          "story_analogy"] if t in ext1["task"].unique()]
    d = ext1.copy()
    d["task_nice"] = d["task"].map(pretty).fillna(d["task"])
    # Ext1 是 pair 对比;3 个 pair 用与主调色相和谐的 3 色(紫/红/蓝)
    palette = [PALETTE["deepseek-reasoner"],
               PALETTE["gpt-5"],
               PALETTE["o4-mini"]]
    sns.barplot(data=d, x="task_nice", y="delta", hue="pair_label", ax=ax,
                order=[pretty[t] for t in order],
                palette=palette, edgecolor="white", linewidth=0.6,
                saturation=0.85)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Δ accuracy  (reasoner − chat counterpart)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("Ext 1 — Reasoning vs Chat: paired family comparison",
                 fontsize=12, loc="left", pad=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", fontsize=9,
              frameon=False, title="")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_ext2_scale_effect(ext2: pd.DataFrame, out: Path) -> None:
    """Ext 2: 同家族 small→full 的 scale 曲线,分 task 子图(2x3 网格)。"""
    setup_style()
    # 只画主 task buckets
    BUCKETS = ["digit_matrix", "letter_string",
               "verbal_ucla_vat", "verbal_sternberg", "verbal_kmiecik",
               "story_analogy"]
    nice = {"digit_matrix": "Digit Matrices",
            "letter_string": "Letter String",
            "verbal_ucla_vat": "Verbal — UCLA VAT",
            "verbal_sternberg": "Verbal — Sternberg",
            "verbal_kmiecik": "Verbal — Kmiecik",
            "story_analogy": "Story Analogies"}
    tasks = [t for t in BUCKETS if t in ext2["task"].unique()]
    families = (ext2.groupby("family")["model_id"].nunique()
                .pipe(lambda s: s[s >= 2]).index.tolist())
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7), sharey=True)
    axes = axes.flatten()
    # 与主调色板的"family 最深色"对齐 — 每个 family 用其 full-size 颜色
    fam_color = {"gpt-4o":      PALETTE["gpt-4o"],
                 "gpt-4.1":     PALETTE["gpt-4.1"],
                 "gpt-5":       PALETTE["gpt-5"],
                 "o-reasoning": PALETTE["o4-mini"],
                 "deepseek":    PALETTE["deepseek-reasoner"]}
    for ax, task in zip(axes, tasks):
        sub = ext2[(ext2["task"] == task) & (ext2["family"].isin(families))]
        for fam, gf in sub.groupby("family"):
            gf = gf.sort_values("size_rank")
            ls = "--" if any(gf["is_reasoning"]) else "-"
            ax.plot(gf["size_rank"], gf["acc"], marker="o", ms=7,
                    linewidth=1.8, label=fam, color=fam_color.get(fam, "#777777"),
                    linestyle=ls)
        ax.set_title(nice.get(task, task), fontsize=10.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["small", "mid", "full"], fontsize=9)
    # 空白 axes 隐藏
    for ax in axes[len(tasks):]:
        ax.axis("off")
    for ax in axes[:len(tasks)]:
        if ax in axes[::ncols]:
            ax.set_ylabel("Accuracy")
    # 全图共用 legend(底部)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
                frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Ext 2 — Scale effect within families (per-task panels)",
                  fontsize=12, x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_ext3_human_similarity(ext3: pd.DataFrame, out: Path) -> None:
    """Ext 3: 4-class profile 与 human 的 r,按 r 降序。"""
    setup_style()
    d = ext3.sort_values("r_vs_human_4class", ascending=False).copy()
    d["display"] = d["model"].map(_short)
    colors = [PALETTE.get(m, "#999999") for m in d["model"]]
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(d["display"], d["r_vs_human_4class"], color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("r  (model vs human, across 4 problem classes)", fontsize=11)
    ax.set_ylim(-0.5, 1.05)
    ax.set_title("Ext 3 — Human similarity on Digit Matrices class profile",
                 fontsize=12, loc="left", pad=10)
    plt.xticks(rotation=30, ha="right", fontsize=10)
    # 数字标注
    for b, v in zip(bars, d["r_vs_human_4class"]):
        ax.text(b.get_x() + b.get_width()/2,
                v + (0.02 if v >= 0 else -0.05),
                f"{v:.2f}", ha="center", fontsize=8,
                color="#222")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_story_panel(by_cond: pd.DataFrame, paper_humans: pd.DataFrame,
                      paper_gpt3: pd.DataFrame, paper_gpt4: pd.DataFrame,
                      out: Path) -> None:
    """X = condition (analogy / similarity),bar = model + human + GPT-3 + GPT-4。"""
    by_cond = by_cond.rename(columns={"subtype": "condition"}).copy()
    extras_df = pd.concat([paper_humans, paper_gpt3, paper_gpt4], ignore_index=True)
    extras_df = extras_df.rename(columns={"agent": "model"})
    full = pd.concat([by_cond[["model", "condition", "acc"]],
                       extras_df[["model", "condition", "acc"]]],
                      ignore_index=True)
    hue_order = _ordered(full)
    plot_grouped_acc(full, x="condition", y="acc", hue="model", out=out,
                     x_order=["similarity", "analogy"], hue_order=hue_order,
                     title="Story Analogies (Gentner et al. 1993): "
                           "similarity vs analogy condition",
                     chance=0.5, ax_size=(14, 5.5))
