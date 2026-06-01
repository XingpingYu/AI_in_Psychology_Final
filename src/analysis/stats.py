"""统计分析:logistic regression、人类相似性 r。"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def acc_by(df: pd.DataFrame, by: Iterable[str]) -> pd.DataFrame:
    g = (df.groupby(list(by))
         .agg(n=("correct", "count"),
              k=("correct", "sum"),
              acc=("correct", "mean"),
              unparseable_rate=("unparseable", "mean"))
         .reset_index())
    g["se"] = (g["acc"] * (1 - g["acc"]) / g["n"]).clip(lower=0).pow(0.5)
    # Wilson 置信区间
    g[["ci_lo", "ci_hi"]] = g.apply(lambda r: pd.Series(_wilson_ci(r["k"], r["n"])),
                                    axis=1)
    return g


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def logistic_main_effects(df: pd.DataFrame,
                          formula: str = "correct ~ C(model) + C(subtype)") -> pd.DataFrame:
    """对单任务的 (correct ~ model + subtype) 跑 logistic regression。
    返回 odds_ratio + 95% CI + p。"""
    d = df.copy()
    d["correct"] = d["correct"].astype(int)
    model = smf.logit(formula, data=d).fit(disp=False)
    params = model.params
    conf = model.conf_int().rename(columns={0: "lo", 1: "hi"})
    pvals = model.pvalues
    out = pd.DataFrame({
        "coef": params,
        "odds_ratio": np.exp(params),
        "or_lo": np.exp(conf["lo"]),
        "or_hi": np.exp(conf["hi"]),
        "p": pvals,
    })
    return out


def human_similarity_r(model_subtype_acc: pd.DataFrame,
                       human_subtype_acc: pd.DataFrame,
                       on: str = "subtype") -> pd.DataFrame:
    """对每个 model 计算与人类在子类型层面的准确率相关 r。
    model_subtype_acc: cols [model, subtype, acc]
    human_subtype_acc: cols [subtype, human_acc]
    """
    merged = model_subtype_acc.merge(human_subtype_acc, on=on, how="inner")
    out = []
    for m, g in merged.groupby("model"):
        if len(g) < 3:
            r = np.nan
        else:
            r = np.corrcoef(g["acc"], g["human_acc"])[0, 1]
        out.append({"model": m, "n_subtypes": len(g), "r_with_human": r})
    return pd.DataFrame(out)
