"""快速探测 DashScope 账户能用的 Qwen 模型。"""
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI, BadRequestError, AuthenticationError, APIError

cli = OpenAI(
    api_key=os.environ["QWEN_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=30,
)

# DashScope 上常见可用模型(按容量从小到大)。
# 闭源商用版(qwen-turbo/plus/max)通常默认有免费额度;
# 开源 2.5/3 instruct 系列常常需要单独开通,因此也列上 -latest 别名。
CANDIDATES = [
    # 闭源商用 — 通常默认可用
    "qwen-turbo", "qwen-turbo-latest",
    "qwen-plus", "qwen-plus-latest",
    "qwen-max", "qwen-max-latest",
    # Qwen2.5 开源 instruct (需开通)
    "qwen2.5-7b-instruct", "qwen2.5-14b-instruct",
    "qwen2.5-32b-instruct", "qwen2.5-72b-instruct",
    "qwen2.5-3b-instruct", "qwen2.5-1.5b-instruct",
    # Qwen3 instruct(更新)
    "qwen3-8b", "qwen3-14b", "qwen3-32b", "qwen3-72b",
    # Reasoning
    "qwq-32b-preview", "qwq-plus", "qwq-32b",
]

print(f"{'model':<30} {'status':<12} {'note'}")
print("-" * 80)
results = []
for m in CANDIDATES:
    try:
        r = cli.chat.completions.create(
            model=m,
            messages=[{"role":"user","content":"Reply 'hi'."}],
            max_tokens=8, temperature=0.0,
        )
        print(f"{m:<30} {'OK':<12} reply={r.choices[0].message.content!r}")
        results.append((m, "OK"))
    except BadRequestError as e:
        msg = str(e)[:120]
        print(f"{m:<30} {'BAD_REQ':<12} {msg}")
        results.append((m, "BAD_REQ"))
    except AuthenticationError as e:
        msg = str(e)[:120]
        print(f"{m:<30} {'AUTH':<12} {msg}")
        results.append((m, "AUTH"))
    except APIError as e:
        msg = str(e)[:120]
        # 403 AccessDenied / 404 Not Found / etc.
        marker = "ACCESS_DENIED" if "AccessDenied" in msg else "API_ERR"
        print(f"{m:<30} {marker:<12} {msg}")
        results.append((m, marker))
    except Exception as e:
        print(f"{m:<30} {'EXC':<12} {repr(e)[:120]}")
        results.append((m, "EXC"))

print("\n=== usable ===")
for m, s in results:
    if s == "OK":
        print(f"  {m}")
