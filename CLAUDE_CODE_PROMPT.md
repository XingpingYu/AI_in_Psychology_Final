# 项目任务:复现并拓展《Emergent analogical reasoning in large language models》

## 背景与目标

我要复现 Webb, Holyoak & Lu (2023, *Nature Human Behaviour*) 的研究
*"Emergent analogical reasoning in large language models"* (DOI: 10.1038/s41562-023-01659-w)。

该论文用认知心理学的**类比推理范式**测试了 GPT-3 (text-davinci-003),
发现它在零样本(zero-shot)条件下达到甚至超过人类大学生水平。原论文的代码与数据在:
https://github.com/taylorwwebb/emergent_analogies_LLM

**我的目标有三层:**
1. **复现核心测试**:复现原文GPT-3实验结果，并用当前较新的模型(尤其是 DeepSeek、GPT-4o/GPT-4.1 系列,以及若干开源模型)
   重跑论文的核心类比任务,与原论文报告的 GPT-3 结果及人类基线对比。
2. **横向比较**:量化不同模型(闭源 vs 开源、不同规模、不同训练范式)之间的差异。
3. **自主拓展**:在复现基础上提出并实现至少 2 个有意义的拓展分析(见下文"拓展方向")，所有基于数据的分析和解读必须打上`[解读]`标签，以与客观实验结果区分开。

请你作为我的研究工程协作者,**分阶段**完成这个项目。每完成一个阶段先暂停、向我汇报进展和初步结果,
再继续下一阶段,不要一次性跑完所有昂贵的 API 调用。

---

## 关键约束与注意事项(请务必先读)

### API key 与成本
- API key 已存放在工作目录下的 `.env` 文件中(`OPENAI_API_KEY`、`DEEPSEEK_API_KEY`)。
  请用 `python-dotenv` 读取,**绝不要把 key 写进代码或日志,也不要 commit `.env`**(确认 `.gitignore` 已包含它)。
- DeepSeek 兼容 OpenAI SDK(`base_url="https://api.deepseek.com"`),可复用同一套调用封装。
- **成本控制是硬性要求**:
  - 先用极小样本(每个子类型 2–3 题)做端到端冒烟测试,确认 pipeline 正确后再放量。
  - 实现**断点续跑 / 结果缓存**:每道题的原始响应写入本地(如 `results/raw/{model}/{task}.jsonl`),
    重跑时跳过已完成的题目。绝不因为脚本崩溃就重刷全部题目。
  - 在 config 里集中管理每个任务的样本量,让我能一键从"冒烟模式"切到"完整模式"。
  - 跑大批量前,先估算并打印预计调用次数和大致 token 量,等我确认。

### 温度与可复现性
- 原论文 `temperature=0`。请默认沿用 `temperature=0` 以保证可复现。
- 把所有调用参数(model name、temperature、max_tokens、seed 如果支持)记录进每个结果文件的元数据。
- 所有代码必须保存下来并形成可直接执行和复现的文件，要有注释。

### 评分标准要严格对齐原论文
- **Digit Matrices(生成式)**:transformation 题答案数字**顺序也要对**才算对;logic 题只要数字**集合**对即可(顺序无所谓)。
- **Digit Matrices / Verbal(多选)**:原论文用 log-prob 打分选答案。
  但 DeepSeek / chat 类模型可能拿不到 token log-prob。**这是一个关键的方法学差异**,请见下文"方法学适配"。
- **Letter string**:生成式,精确匹配目标字符串。
- **Story analogies**:三选一(Story A / Story B / 两者相当),保守地以 50% 为 chance baseline。

### 方法学适配(重要,需要你思考并在报告中讨论)
原论文很多任务依赖 **token log-probabilities** 来给多选题打分。现代 chat-completion 模型(DeepSeek-chat、GPT-4o 等)
通常**不直接暴露每个候选答案的 log-prob**(OpenAI 的 `logprobs` 参数对 chat 模型支持有限,DeepSeek 视情况而定)。
因此请实现两套评测协议,并在代码里做成可切换的策略:

1. **`logprob` 协议**(尽量贴近原论文):对支持 logprobs 的模型/接口使用,逐候选拼接计算平均 log-prob 选最高者。
2. **`prompt` 协议**(现代默认):直接把题目和选项给模型,要求它**只输出选项标号/答案**,
   用稳健的解析器抽取答案。这是大多数现代模型的现实评测方式。

**请明确记录每个模型用的是哪种协议**,因为这本身就是"GPT-3 时代 vs 现在"的一个核心差异,要在报告里讨论它对可比性的影响。

---

## 模型清单

### 必须包含
| 类别 | 模型 | 接入方式 |
|---|---|---|
| 闭源(对照新一代) | OpenAI 当前主力 chat 模型(如 `gpt-4o` / `gpt-4.1`,请先用 OpenAI API 探测我账户可用的模型列表) | OpenAI SDK |
| 闭源(可选,贴近原文) | 若 `text-davinci-003` 仍可用则跑作"忠实复现";若已下线,在报告中说明并以原论文数字为基线 | OpenAI SDK |
| 国产闭源 | DeepSeek-chat、DeepSeek-reasoner(R1 系列,推理模型,重点对比) | DeepSeek SDK(OpenAI 兼容) |

### 开源模型(请帮我选型并给出接入建议)
我希望加入 3–4 个开源模型做对比。**请你先调研并向我建议具体型号**,选型时考虑:
- 覆盖不同规模(如 7B–8B 小模型 vs 32B–70B 中大模型),以便看"规模 vs 类比能力"的关系——
  这正好呼应原论文讨论的"emergent ability"与规模的关系。
- 覆盖不同家族(如 Qwen、Llama、Mistral、Gemma 等)。
- 至少包含一个**推理型(reasoning)开源模型**,与 DeepSeek-reasoner 形成"是否专门做推理训练"的对比维度。

接入方式给我**两个选项**让我选:
- (A) 通过托管推理 API(如 OpenRouter / Together / DeepInfra 等 OpenAI 兼容端点),改动最小;
- (B) 本地用 Ollama / vLLM 拉起小模型(适合 7B–8B,无需额外付费)。

请把"加新模型"设计成只需在 config 里加一条记录即可,**不要在评测逻辑里硬编码任何模型名**。

---

## 任务范围(分优先级)

### 第一优先级(必做,可程序化生成,成本可控)
1. **Digit Matrices(数字矩阵)** —— 论文最核心的任务
   - 规则类型:constant、distribution-of-3、progression(变换题);OR / AND / XOR(逻辑题)。
   - 复现原论文的生成逻辑(单规则到多规则、spatial aligned vs permuted 的逻辑题),并生成干扰项。
   - 优先**直接复用原 GitHub 仓库**里的题目集和干扰项(数据可下载),不要重新发明,除非生成逻辑确有必要重写。
   - 复现分析:按 problem type(one/two/three-rule、logic)、是否含 progression、unique rule 数量、spatial alignment 分组比较。

2. **Letter String Analogies(字母串类比)**
   - 6 种 transformation × 6 种 generalization,含 0–3 个 generalization 的组合,以及 letters→real-world concepts 变体。
   - 复用原仓库题集。复现"准确率随 generalization 数量下降"的曲线,并跨模型比较。

### 第二优先级(做,数据集现成)
3. **Four-term Verbal Analogies(四词语言类比)**
   - 数据集:UCLA VAT、Sternberg & Nigro (1980)、Turney et al. SAT、Jones et al. (2022)。
   - 下载地址见原论文 Data availability(AnalogyInventory.zip 等)。
   - 注意 Jones et al. 的 semantic near/far 分析要保留,这是一个可复现的"人类式语义距离效应"。

4. **Story Analogies(故事类比)**
   - 材料来自 Gentner et al. (1993),含 near / far analogy 两个条件。
   - 这是原论文中**GPT-3 唯一明显输给人类**的任务,且 GPT-4 在此显著更好——
     因此是观察"新模型是否补上了这块短板"的绝佳切入点,**请重点关注**。

### 第三优先级(时间允许再做)
5. **Analogical Problem-Solving(类比问题求解,定性)**
   - Duncker 辐射问题 + 将军故事(Gick & Holyoak 1980)。原文是定性评估。
   - 可设计成:无源故事 / 有源故事 / 加干扰故事 三种条件,看各模型能否被类比引导出"会聚解"。

---

## 拓展方向(请实现至少 2 个,也欢迎你提出更好的)

1. **推理模型 vs 普通模型**:DeepSeek-reasoner、开源 reasoning 模型 vs 普通 chat 模型。
   显式思维链(CoT / reasoning tokens)是否系统性提升类比表现?在哪些题型上提升最大(尤其 far analogy、多规则矩阵)?

2. **规模效应**:在同一家族内(如 Qwen 7B vs 32B vs 72B)看准确率随规模的变化,
   呼应原论文关于"emergent ability"的讨论。

3. **错误模式的人类相似性**:不只比总分,而是复现原论文的"跨子类型准确率相关分析"(模型 vs 人类的 r 值)。
   哪个现代模型的**错误分布**最接近人类?这比"谁分高"更有认知科学意义。

4. **鲁棒性/格式敏感性**:原论文发现 GPT-3 对 prompt 格式敏感(无 prompt、句子格式会掉分)。
   现代模型是否还这么脆弱?对每个模型跑 2–3 种 prompt 变体测稳定性。

5. **零样本 vs 少样本(few-shot)**:原论文强调 zero-shot。可加 1–2 个 in-context 示例,看现代模型增益是否已饱和。

6. **跨语言拓展**(可选,呼应 DeepSeek 的中文能力):把部分语言/故事类比翻译成中文,
   看中英文表现差异——这是原论文没做、且对中文模型特别有意义的方向。

请在动手前**先和我确认你打算实现哪几个拓展**。

---

## 工程要求

### 项目结构(建议,可微调)
```
.
├── .env                    # 已存在,不要 commit
├── .gitignore              # 确保含 .env / results/raw 等
├── config/
│   └── models.yaml         # 模型清单 + 接入参数 + 评测协议(logprob/prompt)
│   └── experiment.yaml     # 各任务样本量、smoke/full 开关、温度等
├── src/
│   ├── clients/            # 统一的 LLM 调用封装(OpenAI/DeepSeek/OpenRouter/Ollama),带重试+缓存
│   ├── tasks/              # 每个任务一个模块:生成/加载题目、构造 prompt、评分
│   ├── eval/               # 跑评测的主循环(支持断点续跑)
│   └── analysis/           # 统计分析 + 画图(对齐论文的 logistic regression / 相关分析)
├── data/                   # 下载/缓存的原始题集
├── results/
│   ├── raw/                # 每题原始响应(jsonl)
│   └── summary/            # 汇总指标(csv)
├── figures/                # 复现 + 拓展的图
├── notebooks/ 或 reports/  # 最终分析报告
└── README.md
```

### 代码质量
- 用 Python。依赖写进 `requirements.txt` 或 `pyproject.toml`。
- 所有 LLM 调用统一走一层封装,内置:指数退避重试、速率限制、超时、原始响应落盘缓存。
- 答案解析器要**稳健**:模型常会输出多余解释,要能从噪声里抽出答案;无法解析的题单独标记 `unparseable`,不要悄悄算错。
- 随机种子固定,题目采样可复现。

### 统计与图表(对齐论文)
- 用 logistic regression(`statsmodels` 或 R 风格)做主效应分析,报告 odds ratio、P 值、95% CI,对齐论文表述。
- 复现论文 Fig.1(四任务总览)、Fig.3(矩阵分题型)、Fig.6(字母串随 generalization)、Fig.7(语言类比)、Fig.8(故事类比)的图,
  但**把人类基线和原论文 GPT-3 数字一并画进去**作为参照(人类/GPT-3 数据可从原论文或其仓库取)。
- 计算每个模型与人类的"跨子类型准确率相关 r",作为人类相似性指标。

### 最终交付
- 一份 Markdown 研究报告(`reports/report.md`),包含:复现结果 vs 原论文、跨模型比较表、拓展分析发现、
  方法学差异讨论(尤其 logprob vs prompt 协议、chat 模型对原范式的适配问题)、局限与后续方向。
- 所有图存到 `figures/`。
- 一个 `README.md` 说明如何复跑(含 smoke / full 模式切换)。

---

## 执行流程(请严格按此分阶段,每阶段结束后暂停汇报)

- **阶段 0 — 勘察与脚手架**:读 `.env`(确认 key 存在,不打印内容)、探测 OpenAI/DeepSeek 可用模型、
  调研开源模型选型给我建议、搭好项目骨架、写好统一 client 封装、跑通对单个模型单道题的"hello world"调用。**暂停,等我确认模型清单和拓展方向。**
- **阶段 1 — Digit Matrices 冒烟测试**:小样本端到端跑通(生成题→调用→评分→汇总),验证评分逻辑正确。**暂停汇报。**
- **阶段 2 — Digit Matrices + Letter String 完整跑**:确认成本预算后放量。**暂停汇报初步结果。**
- **阶段 3 — Verbal + Story analogies**:含数据集下载与缓存。**暂停汇报。**
- **阶段 4 — 拓展分析**:实现我们商定的拓展。**暂停汇报。**
- **阶段 5 — 统计分析、画图、撰写报告**。

每跑一批真实 API 调用前,先打印预计调用量并等我确认。遇到不确定的设计选择(评分细节、协议选择、采样量),
**先问我**,不要擅自做可能影响结果可比性的决定。

现在请从**阶段 0** 开始。
