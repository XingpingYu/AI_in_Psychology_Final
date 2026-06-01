"""v4 主分析:按 Webb 2023 4 个任务划分。
每个任务子节产出:
  1) 总览 csv  (model x task acc + Wilson CI)
  2) 任务专属细分(class / gen_level / condition / dataset / near-far)
  3) 与原文 Human / GPT-3 (有些子任务还有 GPT-4) 的直接对照表
  4) 对应 Webb 论文 Fig.3 / Fig.6 / Fig.7 / Fig.8 风格的画图
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_experiment, load_models
from src.analysis.loader import load_all_raw, attach_problem_class
from src.analysis.stats import acc_by
from src.analysis.extensions import (ext1_reasoning_vs_chat, ext2_scale_effect,
                                     ext3_human_similarity, load_human_class_acc)
from src.analysis import human_baselines as HB
from src.analysis import plots_v4 as P


# 主任务 → Webb 论文中的对应章节
TASK_TO_WEBB_BUCKET = {
    "digit_matrix":      "Digit Matrices",
    "letter_string":     "Letter String Analogies",
    "verbal_ucla_vat":   "Four-term Verbal Analogies",
    "verbal_sternberg":  "Four-term Verbal Analogies",
    "verbal_kmiecik":    "Four-term Verbal Analogies",
    "story_analogy":     "Story Analogies",
}


def main() -> int:
    exp = load_experiment()
    raw_dir = ROOT / exp["paths"]["raw_results_dir"]
    sum_dir = ROOT / exp["paths"]["summary_dir"]
    fig_dir = ROOT / exp["paths"]["figures_dir"]
    sum_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_raw(raw_dir)
    if df.empty:
        print("(no results)"); return 1
    df = attach_problem_class(df)
    df = df[~df["has_error"]].copy()
    main_tasks = set(TASK_TO_WEBB_BUCKET.keys())
    df = df[df["task"].isin(main_tasks)].copy()

    print(f"== {len(df)} records,"
          f" {df['model'].nunique()} models, tasks={sorted(df['task'].unique())} ==\n")

    # =========================== Task 1: Digit Matrices ===========================
    print("### Task 1: Digit Matrices ###")
    dm = df[df["task"] == "digit_matrix"]
    if not dm.empty:
        overall = acc_by(dm, ["model"]).assign(task="digit_matrix")
        by_class = (dm.groupby(["model", "problem_class"])["correct"].mean().reset_index()
                    .rename(columns={"correct": "acc"}))
        by_class.to_csv(sum_dir / "task1_dm_by_class.csv", index=False)
        overall[["model", "n", "k", "acc", "ci_lo", "ci_hi"]].to_csv(
            sum_dir / "task1_dm_overall.csv", index=False)
        # human / gpt3 by class
        human_cls = HB.digit_matrix_human_by_class()
        gpt3_cls  = HB.digit_matrix_gpt3_by_class()
        print("  human by class:", {k: f"{v:.3f}" for k, v in human_cls.items()})
        print("  GPT-3 by class:", {k: f"{v:.3f}" for k, v in gpt3_cls.items()})
        P.plot_digit_matrix_4class(by_class, human_cls, gpt3_cls,
                                   fig_dir / "fig_task1_dm_by_class.png")

    # =========================== Task 2: Letter String ===========================
    print("\n### Task 2: Letter String Analogies ###")
    ls = df[df["task"] == "letter_string"]
    if not ls.empty:
        # gen_level 从原 task items 里 join 出来
        from src.tasks import letter_string as t_ls
        items = t_ls.load_items(n_per_subtype=None)
        meta = {it.item_id: it.meta for it in items}
        ls = ls.copy()
        ls["gen_level"] = ls["item_id"].map(lambda i: (meta.get(i) or {}).get("gen_level", 0))
        by_gen = (ls.groupby(["model", "gen_level"])["correct"].mean().reset_index()
                  .rename(columns={"correct": "acc"}))
        by_gen.to_csv(sum_dir / "task2_ls_by_gen.csv", index=False)
        human_gen = HB.letter_string_human_by_gen()
        print("  human by gen level:", {k: f"{v:.3f}" for k, v in human_gen.items()})
        P.plot_letter_string_by_gen(by_gen, human_gen,
                                     fig_dir / "fig_task2_ls_by_gen.png")

    # =========================== Task 3: Four-term Verbal Analogies ===========================
    print("\n### Task 3: Four-term Verbal Analogies (3 datasets) ###")
    vt = df[df["task"].isin(["verbal_ucla_vat", "verbal_sternberg", "verbal_kmiecik"])]
    if not vt.empty:
        by_task = (vt.groupby(["model", "task"])["correct"].mean().reset_index()
                   .rename(columns={"correct": "acc"}))
        by_task.to_csv(sum_dir / "task3_verbal_by_dataset.csv", index=False)
        # humans / gpt3
        humans = {
            "verbal_ucla_vat": HB.ucla_vat_human_total(),
            "verbal_sternberg": HB.sternberg_paper_baselines()["human"],
            "verbal_kmiecik": HB.kmiecik_paper_baselines()["human"],
        }
        gpt3s = {
            "verbal_ucla_vat": HB.ucla_vat_gpt3_total(),
            "verbal_sternberg": HB.sternberg_paper_baselines()["gpt3"],
            "verbal_kmiecik": HB.kmiecik_paper_baselines()["gpt3"],
        }
        print("  humans:", {k: f"{v:.3f}" for k, v in humans.items()})
        print("  gpt3:  ", {k: f"{v:.3f}" for k, v in gpt3s.items()})
        P.plot_verbal_panel(by_task, humans, gpt3s,
                            fig_dir / "fig_task3_verbal_panel.png")
        # Kmiecik 额外细看 Near/Far × T/F
        kmie = vt[vt["task"] == "verbal_kmiecik"].copy()
        if not kmie.empty:
            # near/far + label 来自 meta
            from src.tasks import verbal_analogy as t_va
            items = t_va.load_items(dataset="kmiecik", n_max=200, stratify=True)
            meta = {it.item_id: (it.meta["near_far"], it.payload["label"]) for it in items}
            kmie["near_far"] = kmie["item_id"].map(lambda i: meta.get(i, (None, None))[0])
            kmie["label"] = kmie["item_id"].map(lambda i: meta.get(i, (None, None))[1])
            by_cell = (kmie.groupby(["model", "near_far", "label"])["correct"]
                       .mean().reset_index().rename(columns={"correct": "acc"}))
            by_cell.to_csv(sum_dir / "task3_kmiecik_near_far_TF.csv", index=False)

    # =========================== Task 4: Story Analogies ===========================
    print("\n### Task 4: Story Analogies ###")
    st = df[df["task"] == "story_analogy"]
    if not st.empty:
        # subtype 就是 condition (analogy/similarity)
        # 关键观察:很多 chat 模型对 'both equally' 题目大量 hedge
        # 我们额外报告:
        #   - strict_acc  : both / unparseable = 错(默认,匹配 chance=50% 论文设定)
        #   - lenient_acc : both 算 0.5(chance);其它一致 - 让"承认不确定"不至于被全惩罚
        #   - commit_rate : 1 - both 率,反映承诺度
        from src.utils.io import jsonl_read
        rows = []
        for model_dir in raw_dir.iterdir():
            if not model_dir.is_dir(): continue
            jf = model_dir / "story_analogy.jsonl"
            if not jf.exists(): continue
            for r in jsonl_read(jf):
                rows.append({
                    "model": model_dir.name,
                    "condition": r.get("subtype"),
                    "correct_strict": bool(r.get("correct")),
                    "parsed": r.get("parsed"),
                    "is_both": (r.get("parsed") == "both"),
                    "unparseable": bool(r.get("unparseable")),
                })
        story_df = pd.DataFrame(rows)
        # strict (= correct as scored)
        by_cond_strict = (story_df.groupby(["model", "condition"])["correct_strict"]
                          .mean().reset_index()
                          .rename(columns={"correct_strict": "acc"}))
        # lenient: both 算 0.5
        def lenient_row(r):
            if r["is_both"]:
                return 0.5
            return 1.0 if r["correct_strict"] else 0.0
        story_df["lenient"] = story_df.apply(lenient_row, axis=1)
        by_cond_lenient = (story_df.groupby(["model", "condition"])["lenient"]
                           .mean().reset_index()
                           .rename(columns={"lenient": "acc"}))
        # commit rate
        commit = (story_df.assign(committed=~story_df["is_both"])
                  .groupby(["model", "condition"])["committed"].mean()
                  .reset_index()
                  .rename(columns={"committed": "commit_rate"}))
        by_cond = by_cond_strict.copy()
        by_cond["lenient_acc"] = by_cond_lenient["acc"].values
        by_cond["commit_rate"] = commit["commit_rate"].values
        by_cond.to_csv(sum_dir / "task4_story_by_condition.csv", index=False)
        # 总览
        total = (st.groupby("model")["correct"].mean().reset_index()
                 .rename(columns={"correct": "acc"}))
        total.to_csv(sum_dir / "task4_story_total.csv", index=False)
        print(f"  story by condition (strict / lenient / commit_rate):")
        print(by_cond.round(3).to_string(index=False))
        # paper baselines
        paper_h = HB.story_baselines_by_condition()
        paper_h = paper_h[paper_h["agent"] == "Human (paper)"]
        paper_g3 = HB.story_baselines_by_condition()
        paper_g3 = paper_g3[paper_g3["agent"] == "GPT-3 (paper)"]
        paper_g4 = HB.gpt4_story_baselines_by_condition()
        P.plot_story_panel(by_cond.rename(columns={"condition": "subtype"}),
                           paper_h, paper_g3, paper_g4,
                           fig_dir / "fig_task4_story_panel.png")
        # 新增:hedging vs accuracy 散点
        P.plot_story_commit_vs_acc(by_cond,
                                    fig_dir / "fig_task4_story_commit_vs_acc.png")
        print("  human/gpt-3/gpt-4 by condition (paper):")
        for d in [HB.story_baselines_by_condition(),
                  HB.gpt4_story_baselines_by_condition()]:
            print(d.to_string(index=False))

    # =========================== Overall summary across 4 tasks ===========================
    print("\n### Overall summary across 4 tasks ###")
    # 把 verbal 三个数据集均值作为 verbal 总分,story 和其它直接对应
    df["bucket"] = df["task"].map(TASK_TO_WEBB_BUCKET)
    by_bucket = (df.groupby(["model", "bucket"])["correct"].mean().reset_index()
                 .rename(columns={"correct": "acc", "bucket": "task"}))
    by_bucket.to_csv(sum_dir / "task_overall_by_bucket.csv", index=False)

    human_per_bucket = {
        "Digit Matrices": float(np.mean(list(HB.digit_matrix_human_by_class().values()))),
        "Letter String Analogies": float(np.mean(list(HB.letter_string_human_by_gen().values()))),
        "Four-term Verbal Analogies": HB.ucla_vat_human_total(),
        "Story Analogies": float(HB.story_baselines_by_condition()
                                  [HB.story_baselines_by_condition()["agent"] == "Human (paper)"]
                                  ["acc"].mean()),
    }
    gpt3_per_bucket = {
        "Digit Matrices": float(np.mean(list(HB.digit_matrix_gpt3_by_class().values()))),
        "Four-term Verbal Analogies": HB.ucla_vat_gpt3_total(),
        "Story Analogies": float(HB.story_baselines_by_condition()
                                  [HB.story_baselines_by_condition()["agent"] == "GPT-3 (paper)"]
                                  ["acc"].mean()),
        "Letter String Analogies": float("nan"),  # GPT-3 letter-string per-class not extracted
    }
    P.plot_overall_with_humans(by_bucket, human_per_bucket, gpt3_per_bucket,
                               fig_dir / "fig_overall_4tasks.png")

    # =========================== Ext 1 / 2 / 3 ===========================
    print("\n### Extensions 1 / 2 / 3 ###")
    models = load_models()
    mm = pd.DataFrame([{"id": m["id"], "family": m["family"],
                        "reasoning": bool(m.get("reasoning", False)),
                        "size_b": m.get("size_b")}
                       for m in models if m.get("enabled", False)])
    mm = mm[mm["id"].isin(df["model"].unique())].copy()

    # Ext 1
    e1 = ext1_reasoning_vs_chat(df, mm)
    if not e1.empty:
        e1.to_csv(sum_dir / "ext1_reasoning_vs_chat.csv", index=False)
        P.plot_ext1_reasoning_vs_chat(e1, fig_dir / "fig_ext1_reasoning_vs_chat.png")
        print(f"  Ext1 saved")

    # Ext 2
    e2 = ext2_scale_effect(df, mm)
    if not e2.empty:
        e2.to_csv(sum_dir / "ext2_scale_effect.csv", index=False)
        P.plot_ext2_scale_effect(e2, fig_dir / "fig_ext2_scale_effect.png")
        print(f"  Ext2 saved")

    # Ext 3
    e3 = ext3_human_similarity(df, load_human_class_acc())
    if not e3.empty:
        e3.to_csv(sum_dir / "ext3_human_similarity.csv", index=False)
        P.plot_ext3_human_similarity(e3, fig_dir / "fig_ext3_human_similarity.png")
        print(f"  Ext3 saved")

    print(f"\nSaved to {sum_dir} and {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
