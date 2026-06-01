"""检查原仓库里人类数据的具体结构,看能不能用来做严格的 model-vs-human 对比。"""
import numpy as np
from pathlib import Path

REPO = Path("data/repo_original")

print("=== letter_string human prob_subtype_acc ===")
d = np.load(REPO/"letter_string/behavioral_results/prob_subtype_acc.npz",
            allow_pickle=True)
for k in d.keys():
    v = d[k]
    print(f"  {k}: dtype={v.dtype} shape={v.shape}")
    if v.ndim == 0 and v.dtype == object:
        inner = v.item()
        if isinstance(inner, dict):
            ks = list(inner.keys())
            print(f"    dict keys (first 10 of {len(ks)}): {ks[:10]}")
            sk = ks[0]
            print(f"    inner[{sk!r}] type={type(inner[sk]).__name__} "
                  f"shape={getattr(inner[sk], 'shape', 'N/A')}")

print("\n=== digit_mat exp1 probcat_gen_acc_behavior ===")
d2 = np.load(REPO/"digit_mat/exp1_behavioral_data/probcat_gen_acc_behavior.npz",
             allow_pickle=True)
for k in d2.keys():
    v = d2[k]
    print(f"  {k}: dtype={v.dtype} shape={v.shape}")
    if v.ndim == 0 and v.dtype == object:
        inner = v.item()
        print(f"    item type: {type(inner).__name__}")
        if isinstance(inner, dict):
            ks = list(inner.keys())
            print(f"    dict keys: {ks}")
            for sk in ks[:3]:
                val = inner[sk]
                print(f"    [{sk!r}]: type={type(val).__name__} "
                      f"shape={getattr(val,'shape','N/A')} value_preview={str(val)[:80]}")

print("\n=== UCLA_VAT human individual subject data ===")
import pandas as pd
xls = pd.ExcelFile(REPO/"UCLA_VAT/UCLA_VAT_ind_subj_data.xlsx")
print("sheets:", xls.sheet_names)
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
print("shape:", df.shape)
print("cols:", list(df.columns)[:15])
print(df.head(3))
