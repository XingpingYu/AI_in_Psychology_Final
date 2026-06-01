"""一次性查看原仓库的 npz 数据结构。"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "data" / "repo_original"

print("=== digit_mat: all_problems.npz ===")
d = np.load(REPO / "digit_mat" / "all_problems.npz", allow_pickle=True)
print("keys:", list(d.keys()))
all_prob = d["all_problems"].item()
print("problem types:", list(all_prob.keys()))
for pt, body in all_prob.items():
    perm = body["perm_invariant"]
    prob = body["prob"]
    answers = body["answer_choices"]
    correct_ind = body["correct_ind"]
    print(f"  {pt:30s} N={len(prob)} perm_inv={perm} "
          f"prob.shape={prob.shape} ans.shape={answers.shape}")
    if pt == list(all_prob.keys())[0]:
        print("   example prob[0]:", prob[0])
        print("   example answer_choices[0]:", answers[0])
        print("   example correct_ind[0]:", correct_ind[0])

print("\n=== digit_mat: all_problems_1thru5.npz ===")
d2 = np.load(REPO / "digit_mat" / "all_problems_1thru5.npz", allow_pickle=True)
print("keys:", list(d2.keys()))
ap2 = d2["all_problems"].item()
print("problem types:", list(ap2.keys()))

print("\n=== letter_string: all_prob.npz ===")
d3 = np.load(REPO / "letter_string" / "all_prob.npz", allow_pickle=True)
print("keys:", list(d3.keys()))
ap3 = d3["all_prob"].item()
print("problem types (first 5 of", len(ap3), "):", list(ap3.keys())[:5])
first = list(ap3.keys())[0]
print("example body keys:", list(ap3[first].keys()))
print("example prob[0]:", ap3[first]["prob"][0])
print("example correct[0]:", ap3[first].get("correct", "N/A")[0]
      if "correct" in ap3[first] else "[no correct field]")

print("\n=== story_analogies dir contents (csv only since no npz) ===")
for f in (REPO / "story_analogies").glob("*.csv"):
    import pandas as pd
    df = pd.read_csv(f)
    print(f, "shape", df.shape, "cols", list(df.columns)[:8])

print("\n=== UCLA_VAT xlsx ===")
import pandas as pd
xls = pd.ExcelFile(REPO / "UCLA_VAT" / "UCLA_VAT.xlsx")
print("sheets:", xls.sheet_names)
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
print("first sheet shape:", df.shape, "cols:", list(df.columns)[:10])
print(df.head(3))
