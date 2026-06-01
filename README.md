# Emergent Analogical Reasoning — 复现与拓展

复现 **Webb, Holyoak & Lu (2023)** *"Emergent analogical reasoning in large language models"* (**Nature Human Behaviour**, DOI: [10.1038/s41562-023-01659-w](https://doi.org/10.1038/s41562-023-01659-w)),并把原文 4 大类比任务扩展到 **13 个现代 LLM** + 3 个额外 verbal 数据集。

原文仓库:[`taylorwwebb/emergent_analogies_LLM`](https://github.com/taylorwwebb/emergent_analogies_LLM)

---

## TL;DR

- **13 模型 × 4 任务 × 6 数据集 = 13,610 trial-level 记录**
- **每章节明示数据与原文的同源关系**:有的逐题可比,有的只能比聚合数字
- 主要发现:
  - Digit Matrices:**6/13 模型 ≥ 0.93,远超人类 0.58**;`o3-mini` 拿头名 0.95
  - Letter String:`gpt-5` / `o4-mini` 接近人类,但 gen-level=3 上 ≤0.31 仍是真正瓶颈
  - Verbal:UCLA VAT / Sternberg 已 saturated;Kmiecik (Jones et al.) 仍能拉开差距
  - **Story Analogies(v4 核心新发现)**:多数 chat 模型大量 hedging 到 "both equally";`gpt-5` 系列 commit 率 76% 是仅有接近人类的两个模型

完整报告见 [`reports/report.md`](reports/report.md);设计决策细节见报告 §8b "Methodology Decisions"。

---

## 文件结构

```
.
├── README.md                # 本文件
├── requirements.txt         # Python 依赖
├── .env.example             # API key 模板(实际 .env 在 .gitignore)
├── .gitignore
│
├── config/
│   ├── models.yaml          # 模型清单(13 entries + Qwen 占位)
│   └── experiment.yaml      # 任务样本量 + smoke/full 开关 + 调用参数
│
├── docs/
│   └── s41562-023-01659-w.pdf   # 原论文 PDF
│
├── data/                    # ★ 不纳入 git;运行前需自行下载,见下方「数据下载」
│   ├── repo_original/       # clone: taylorwwebb/emergent_analogies_LLM
│   │   ├── digit_mat/       # 32 子类型 + 人类 + GPT-3 baseline
│   │   ├── letter_string/   # 28 子类型 + behavioral_results/
│   │   ├── UCLA_VAT/        # 80 verbal analogies + 个体被试数据
│   │   └── story_analogies/ # 1044 行 human/GPT-3 csv + 72 行 GPT-4 csv
│   ├── AnalogyInventory/    # 解压自 UCLA cvl lab AnalogyInventory.zip
│   │   └── Public/
│   │       ├── Cognitive Psychology.xlsx  # 含 Sternberg / Kmiecik / Rattermann sheet
│   │       └── ...
│   └── AnalogyInventory.zip
│
├── src/                     # 核心库(可作为 Python package import)
│   ├── clients/             # 统一 LLM 客户端(OpenAI / DeepSeek / DashScope / OpenRouter)
│   │   ├── base.py          # BaseLLMClient + LLMResponse + 重试 + 缓存
│   │   ├── openai_client.py # OpenAI 兼容 API(支持 chat + completion endpoint)
│   │   └── registry.py      # 按 provider 字段路由
│   ├── tasks/               # 4 个 Webb 任务模块
│   │   ├── base.py          # TaskItem / ParsedAnswer
│   │   ├── digit_matrix.py
│   │   ├── letter_string.py
│   │   ├── verbal_analogy.py# UCLA VAT + Sternberg + Kmiecik(共用 format/parse/score)
│   │   └── story_analogy.py # Rattermann sheet
│   ├── eval/runner.py       # 评测主循环(并发 + 断点续跑 + completion-style 路由)
│   ├── analysis/
│   │   ├── loader.py        # 把 raw jsonl 汇总为长表
│   │   ├── stats.py         # Wilson CI + logistic regression
│   │   ├── extensions.py    # Ext1/2/3 计算
│   │   ├── human_baselines.py # 4 任务的 human / GPT-3 / GPT-4 基线提取
│   │   └── plots_v4.py      # 全部 figures 风格 + 调色板
│   └── utils/               # config / env / io / logging
│
├── scripts/                 # 命令行入口
│   ├── phase0_probe.py      # 启动自检 — 探测 .env + 列模型
│   ├── phase0b_probe_openai.py  # 全模型 hello-world
│   ├── probe_dashscope.py   # 探阿里云 Qwen 模型可用性(占位,需开通账户)
│   ├── run_eval.py          # 主评测入口
│   ├── reparse.py           # Reasoner 截断兜底,从 reasoning_text 救回答案
│   ├── analyze_v4.py        # 主分析 — 按 Webb 4 任务出图与表
│   ├── case_studies.py      # 从 raw 挖代表性 case 写成 markdown
│   ├── estimate_cost.py     # 跑前算总 token 与 $
│   └── inspect_*.py         # 三个数据探查辅助
│
├── tests/                   # 离线自检(零 API)
│   ├── test_smoke.py        # 模块 import 是否完整
│   ├── test_tasks_offline.py # 三个核心 task 的 load/format/parse/score
│   └── test_new_tasks.py    # v4 新增 task 的同上
│
├── results/
│   ├── raw/                 # 每模型每任务一个 jsonl;断点续跑文件源
│   │   └── {model_id}/
│   │       ├── digit_matrix.jsonl
│   │       ├── letter_string.jsonl
│   │       ├── verbal_ucla_vat.jsonl
│   │       ├── verbal_sternberg.jsonl
│   │       ├── verbal_kmiecik.jsonl
│   │       └── story_analogy.jsonl
│   └── summary/             # 汇总 csv(由 analyze_v4 产出)
│       ├── task1_dm_*.csv
│       ├── task2_ls_by_gen.csv
│       ├── task3_*.csv
│       ├── task4_story_*.csv
│       ├── task_overall_by_bucket.csv
│       └── ext{1,2,3}_*.csv
│
├── figures/                 # 全部 9 张 v4 figures(报告 inline 引用)
│   ├── fig_overall_4tasks.png       # 4 任务总览
│   ├── fig_task1_dm_by_class.png    # Digit Matrices by class
│   ├── fig_task2_ls_by_gen.png      # Letter String by gen level
│   ├── fig_task3_verbal_panel.png   # Verbal 3 datasets panel
│   ├── fig_task4_story_panel.png    # Story by condition
│   ├── fig_task4_story_commit_vs_acc.png  # ★ hedging 散点(v4 独有)
│   ├── fig_ext1_reasoning_vs_chat.png
│   ├── fig_ext2_scale_effect.png    # 2x3 task panels
│   └── fig_ext3_human_similarity.png
│
├── reports/
│   ├── report.md            # 主报告(按 Webb 4 任务结构,含 §8b 设计 tips)
│   └── case_studies.md      # 5 个代表性 case
│
├── logs/                    # 历次运行日志(*.log,不进 git)
└── cache/                   # 客户端响应缓存(jsonl,不进 git)
```

---

## 环境

- **Python 3.11**(其他 3.10+ 也应该能跑,但只测了 3.11)
- 用 conda 创建独立环境;**所有大文件路径都避开 C: 盘**

```bash
conda create -n analogy_llm python=3.11 -y
conda activate analogy_llm
pip install -r requirements.txt
```

主要依赖(详细见 [requirements.txt](requirements.txt)):
`openai>=1.40`,`python-dotenv`,`pyyaml`,`pandas`,`numpy`,`scipy`,`matplotlib`,`seaborn`,`statsmodels`,`tqdm`,`tenacity`,`pypdf`(只用于 docs/ 检索),`openpyxl`。

---

## 数据下载

> **本仓库不附带 `data/` 下的数据文件**(均为第三方原始素材,体积较大且各有归属许可,详见文末「数据归属」)。
> 克隆本仓库后,`data/` 只有一个占位 `.gitkeep`;请按下面两步把数据放到位,代码默认从 `data/` 读取(`config/experiment.yaml` 中 `paths.data_dir: data`)。

**① Webb 原仓库数据**(Digit Matrices / Letter String / UCLA VAT / Story Analogies 题集 + 人类/GPT-3/GPT-4 基线)

直接把原仓库克隆到 `data/repo_original`:

```bash
git clone https://github.com/taylorwwebb/emergent_analogies_LLM.git data/repo_original
```

**② UCLA AnalogyInventory**(Sternberg 1980 / Kmiecik 2021 / Gentner-Rattermann-Forbus 1993 stimuli)

下载 zip 后解压到 `data/AnalogyInventory`(解压后会自动得到 `data/AnalogyInventory/Public/*.xlsx`):

```bash
# Linux / macOS
curl -L -o data/AnalogyInventory.zip http://cvl.psych.ucla.edu/resources/AnalogyInventory.zip
unzip data/AnalogyInventory.zip -d data/AnalogyInventory
```

```powershell
# Windows PowerShell
Invoke-WebRequest -Uri "http://cvl.psych.ucla.edu/resources/AnalogyInventory.zip" -OutFile "data/AnalogyInventory.zip"
Expand-Archive -Path "data/AnalogyInventory.zip" -DestinationPath "data/AnalogyInventory" -Force
```

放好后 `data/` 应当是这个结构:

```
data/
├── repo_original/                    # ① clone 得到
│   ├── digit_mat/all_problems.npz
│   ├── letter_string/all_prob.npz
│   ├── UCLA_VAT/UCLA_VAT.xlsx
│   └── story_analogies/
├── AnalogyInventory/                 # ② 解压得到
│   └── Public/
│       ├── Cognitive Psychology.xlsx
│       └── ...
└── AnalogyInventory.zip              # ② 下载得到(可删)
```

**验证数据就位**:跑离线自检(见下方「离线自检」),三个测试会从上述路径 load/parse,任何缺文件都会立刻报错。

---

## 配置 API key

把 [`.env.example`](.env.example) 复制成 `.env`,填入:

```ini
# 必需
OPENAI_API_KEY=sk-proj-...        # 项目密钥或服务密钥
DEEPSEEK_API_KEY=sk-...

# 可选 — 第三方代理(若 OPENAI_API_KEY 是代理)
OPENAI_BASE_URL=https://api.your-proxy.com/v1

# 可选 — 阿里云 DashScope 的 Qwen 系列(目前 models.yaml 未启用)
QWEN_API_KEY=sk-...

# 可选 — OpenRouter 多家开源模型聚合
OPENROUTER_API_KEY=sk-or-...
```

> **永远不要把 `.env` commit 进 git;[.gitignore](.gitignore) 已默认排除。**

---

## 一键复现 v4 全量(约 3 小时 / $11)

```bash
# 1. 启动自检 — 列出账户能用的模型 + 跑全模型 hello-world(成本 ≈ $0.01)
python scripts/phase0_probe.py
python scripts/phase0b_probe_openai.py

# 2. (可选)dry-run 估算成本
python scripts/run_eval.py \
  --tasks digit_matrix letter_string verbal_ucla_vat verbal_sternberg verbal_kmiecik story_analogy \
  --mode full --dry-run
python scripts/estimate_cost.py

# 3. 主跑(后台运行,断点续跑;若中断重跑会自动跳过已完成题目)
python scripts/run_eval.py \
  --tasks digit_matrix letter_string verbal_ucla_vat verbal_sternberg verbal_kmiecik story_analogy \
  --mode full --concurrency 4

# 4. Reasoner 截断兜底(从 reasoning_text 末尾抽答案;不再花 API)
python scripts/reparse.py

# 5. 主分析 — 出图、出表
python scripts/analyze_v4.py

# 6. (可选)挖代表性 case 写 markdown
python scripts/case_studies.py
```

跑完后:
- 9 张 figures 在 [`figures/`](figures/)
- 13 个汇总 csv 在 [`results/summary/`](results/summary/)
- 主报告自动渲染到 [`reports/report.md`](reports/report.md)(其中已 inline 引用所有 figures)

---

## 单任务 / 单模型选跑

只跑一个模型的一个任务(便于调试):
```bash
python scripts/run_eval.py --tasks digit_matrix --models deepseek-chat --mode smoke
```

切 smoke ↔ full 的方式:
- 命令行 `--mode smoke|full`
- 环境变量 `EXP_MODE=full`
- 或改 [`config/experiment.yaml`](config/experiment.yaml) 里的 `mode:` 字段

smoke 默认每子类型 2 题、verbal 4 题、story 12 trial,用于 pipeline 自检(总成本 ~$0.02)。

---

## 任务 — 数据 — 与原文的对应关系

| Webb 论文章节 | 任务 | 数据集 | 题数 | 同源? | 评分 |
|---|---|---|---:|:-:|---|
| §Exp 1 | **Digit Matrices** | `data/repo_original/digit_mat/all_problems.npz` | 314(32 子类型 × 10) | **是** | generative,顺序/集合匹配 |
| §Exp 2 | **Letter String** | `data/repo_original/letter_string/all_prob.npz` | 224(28 子类型 × 8) | **是** | generative,字符串完全匹配 |
| §Verbal | UCLA VAT | `data/repo_original/UCLA_VAT/UCLA_VAT.xlsx` | 80(全) | **是** | 2-AFC,prompt 协议 |
| §Verbal | Sternberg 1980 | `data/AnalogyInventory/Public/Cognitive Psychology.xlsx::Sternberg` | 197(全) | 同源数据,无 per-item baseline | 2-AFC |
| §Verbal | Kmiecik (= Jones 2022) | 同上 `::Kmiecik` | 160(分层 Near/Far × T/F 各 40) | 同上 | yes/no 判断 |
| §Story | Gentner 1993 stimuli | 同上 `::Rattermann` | 72(18 × 4 trial) | **是,Webb Data Availability 明确指向** | 三选一(A/B/both) |

---

## 模型清单(13 enabled + Qwen 占位)

完整定义见 [`config/models.yaml`](config/models.yaml)。

| Family | Models | provider | reasoning | 备注 |
|---|---|---|:-:|---|
| **GPT-3 时代代理** | `gpt-3.5-turbo-instruct` | OpenAI completions | ❌ | text-davinci-003 已下线;**走原文 completion-style prompt + completion API** |
| gpt-4o | `gpt-4o-mini`, `gpt-4o` | OpenAI chat | ❌ | |
| gpt-4.1(规模 3 档) | `gpt-4.1-nano`, `gpt-4.1-mini`, `gpt-4.1` | OpenAI chat | ❌ | scale effect 主对照 |
| gpt-5(规模 3 档) | `gpt-5-nano`, `gpt-5-mini`, `gpt-5` | OpenAI chat | (隐式) | scale effect 主对照;`temperature=1` 强制 |
| o-reasoning | `o3-mini`, `o4-mini` | OpenAI chat | ✅ | OpenAI 显式 reasoning,`max_completion_tokens=16384` |
| deepseek | `deepseek-chat`, `deepseek-reasoner` | DeepSeek | ❌ / ✅ | 国产对照 + reasoning 同家族对 |
| **(占位)** qwen2.5 系列 | 7B / 14B / 32B / 72B | DashScope | ❌ | 阿里云账户未开通,目前 disabled |

### 加新模型

只需在 `config/models.yaml` 加一条 entry,**不用改任何评测代码**:

```yaml
- id: my-new-model
  provider: openai        # 或 deepseek / openrouter / dashscope
  model: actual-model-name-for-the-API
  family: my-family       # 用于 Ext1 / Ext2 分析
  size_b: null            # 开源模型填 B
  reasoning: false        # true 启用 reasoning_content 提取与额外 token 预算
  temperature: 0.0
  max_tokens: 1024
  completion_style: false # true 走 legacy /v1/completions(gpt-3.5-instruct 类)
  enabled: true
```

---

## 设计取舍(快速版,详细在报告 §8b)

| 维度 | 决策 | 为什么 |
|---|---|---|
| 评测协议 | prompt 协议(chat 模型) + completion-style(gpt-3.5-inst) | chat API 不暴露 forced-completion logprob |
| 样本量 | digit 10/类(原 40)、letter 8/类(原 50)、Kmiecik 160/720 | reasoner ~11s/题,reasoning 模型时长瓶颈 |
| 并发 | per-model 4 worker | OpenAI / DeepSeek 安全 RPS |
| 缓存 | 3 层(item-level jsonl + 客户端响应 cache + reparse 兜底) | 断点续跑 + 不重复花钱 |
| Reasoner 截断 | max_tokens 给到 6144 / 16384;reasoning_text 末尾抽答案 | thinking 偶尔吃光预算 |
| Story 评分 | 同报 strict / lenient / commit_rate | RLHF 模型大量 hedge "both equally",单一指标不够 |
| 2-AFC 位置 | item_id 末位奇偶轮换 D/D′ 位置 | 消位置偏置 + 可复现 |
| 图色 | 5 family 各一个 hue,内部 light→dark = small→full;reference 灰阶 | 13+ 模型避免 rainbow,接近 Nature 风格 |

---

## 离线自检(零 API)

每次改了 task / parser / scorer 后,先跑:
```bash
python tests/test_smoke.py            # 模块 import + config 解析
python tests/test_tasks_offline.py    # 三个核心 task 的 golden-answer 测试
python tests/test_new_tasks.py        # v4 新增任务的同上
```
3 个测试全过约 1 秒,不打 API。

---

## 引用

原论文:

```bibtex
@article{webb2023emergent,
  title={Emergent analogical reasoning in large language models},
  author={Webb, Taylor and Holyoak, Keith J and Lu, Hongjing},
  journal={Nature Human Behaviour},
  volume={7},
  number={9},
  pages={1526--1541},
  year={2023},
  doi={10.1038/s41562-023-01659-w}
}
```

---

## 数据归属

- Digit Matrices / Letter String / UCLA VAT / Story Analogies 题集 + 人类 / GPT-3 / GPT-4 聚合基线 → Webb, Holyoak & Lu 2023 仓库(BSD-style)
- Sternberg 1980 / Kmiecik 2021 / Gentner-Rattermann-Forbus 1993 stimuli → [UCLA Cognitive Vision Lab AnalogyInventory.zip](http://cvl.psych.ucla.edu/resources/AnalogyInventory.zip)
- 本项目代码 **MIT**
