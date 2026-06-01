# Case Studies (从 8,033 条 raw 响应里挑选)

本文档由 `scripts/case_studies.py` 自动生成,展示模型在不同任务上的代表性成败 case,用于补充报告 §11。

### Case A: completion-style bias — `gpt-3.5-turbo-instruct` 把单格补成整行

`item_id = row_constant__0003`
**Subtype**: `row_constant`

**Prompt(模型实际看到的)**:
```
You are solving a 3x3 matrix reasoning problem. Each cell of the matrix is a set of digits enclosed in square brackets. The bottom-right cell is missing (marked as [?]). Find the cell that completes the matrix following the same pattern as the other rows/columns.

[5] [5] [5]
[7] [7] [7]
[3] [3] [?]

Reply with ONLY the missing cell on a single line in the format [d1 d2 ...]. Do not include any explanation.
```

**正确答案**:`[3]`

**各模型响应**:

| Model | Correct? | Parsed | Response (head) |
|---|:-:|---|---|
| gpt-3.5-turbo-instruct | ❌ | `[3, 3, 3]` | [3 3 3] |
| gpt-4o-mini | ✅ | `[3]` | [3] |
| gpt-4.1 | ✅ | `[3]` | [3] |
| gpt-5 | ✅ | `[3]` | [3] |
| o3-mini | ✅ | `[3]` | [3] |
| deepseek-chat | ✅ | `[3]` | [3] |
| deepseek-reasoner | ✅ | `[3]` | [3] |

**[解读]** 单格答案在 chat 模型上都没问题,而 `gpt-3.5-turbo-instruct` 把 `[6] [6] [?]` 补成 `[6 6 6]` 整行 — completion 模型对格式化模板的自动补全先验击败了我们的 `reply only the missing cell` 指令。要让它正确答题需要走原文 logprob 协议,而非 chat-style prompt。

### Case B: Letter String gen-3:多数模型失败的真正难题

`item_id = fix_alphabet__003`
**Subtype**: `fix_alphabet`

**Prompt(模型实际看到的)**:
```
Let's try to complete the pattern:

[e f g h j] [e f g h i]
[s w x y z] [?]

Reply with ONLY the missing target on a single line in the same [t1 t2 ...] format. Do not include any explanation.
```

**正确答案**:`['v', 'w', 'x', 'y', 'z']`

**各模型响应**:

| Model | Correct? | Parsed | Response (head) |
|---|:-:|---|---|
| gpt-3.5-turbo-instruct | ❌ | `['a']` | [s w x y z] [a] |
| gpt-4o-mini | ❌ | `['a', 'b', 'c', 'd', 'e']` | [a b c d e] |
| gpt-4o | ❌ | `['s', 'w', 'x', 'y', 'y']` | [s w x y y] |
| gpt-4.1-nano | ❌ | `['s', 'w', 'x', 'y', 'z']` | [s w x y z] |
| gpt-4.1-mini | ❌ | `['s', 'w', 'x', 'y', 'a']` | [s w x y a] |
| gpt-4.1 | ❌ | `['s', 'w', 'x', 'y', 'a']` | [s w x y a] |
| gpt-5-nano | ❌ | `['s', 'w', 'x', 'y', 'y']` | [s w x y y] |
| gpt-5-mini | ❌ | `['s', 'w', 'x', 'y', 'y']` | [s w x y y] |
| gpt-5 | ✅ | `['v', 'w', 'x', 'y', 'z']` | [v w x y z] |
| o3-mini | ❌ | `['s', 'w', 'x', 'y', 'y']` | [s w x y y] |
| o4-mini | ❌ | `['s', 't', 'u', 'v', 'w']` | [s t u v w] |
| deepseek-chat | ❌ | `['s', 'w', 'x', 'y', 'z']` | [s w x y z] |
| deepseek-reasoner | ❌ | `None` | [thinking 截断,无 response_text] |

**deepseek-reasoner 的 thinking 片段**:
```
We are given two sets: [e f g h j] and [e f g h i]. Then [s w x y z] and we need the missing pattern. Observing the first pair: first is e f g h j (missing i), second is e f g h i (missing j). They are consecutive letters but with one missing. So the pattern might be that the second set is the same 
…(省略中间 thinking)…
 letters. So the "missing" pattern is different: one missing one, the other missing three. That suggests maybe the pattern is that the number of missing letters increases? But then the second given set is [s w x y z] which misses three, so the target
```

**[解读]** 这是一道 `fix alphabetic order` — src_A 的最后一个字符 `j` 不连续,要改成 `i` 让序列连续;src_B 的首字符 `s` 不连续 (与 `w` 不衔接),要改成 `v`,得到 `[v w x y z]`。**13 个模型里只有 gpt-5 答对**;其他大多数模型采用了'局部修改最后一个'的浅启发式 (`[s w x y a]`/`[s w x y y]`/`[s w x y z]`),没有意识到出错的元素其实在序列开头。这是 emergent analogy 'far transfer' 上的真实瓶颈。

### Case C: Reasoning 'rescue':`deepseek-chat` 错,`deepseek-reasoner` 对

`item_id = three_rule_comb1__0002`
**Subtype**: `three_rule_comb1`

**Prompt(模型实际看到的)**:
```
You are solving a 3x3 matrix reasoning problem. Each cell of the matrix is a set of digits enclosed in square brackets. The bottom-right cell is missing (marked as [?]). Find the cell that completes the matrix following the same pattern as the other rows/columns.

[8 6 5] [3 8 0] [4 3 2]
[3 6 5] [4 8 0] [8 3 2]
[4 6 5] [8 8 0] [?]

Reply with ONLY the missing cell on a single line in the format [d1 d2 ...]. Do not include any explanation.
```

**正确答案**:`[3, 3, 2]`

**各模型响应**:

| Model | Correct? | Parsed | Response (head) |
|---|:-:|---|---|
| deepseek-chat | ❌ | `[2, 8, 0]` | [2 8 0] |
| deepseek-reasoner | ✅ | `[3, 3, 2]` | [3 3 2] |
| gpt-4o | ❌ | `[2, 3, 2]` | [2 3 2] |
| o4-mini | ✅ | `[3, 3, 2]` | [3 3 2] |

**deepseek-reasoner 的 thinking 片段**:
```
We are given a 3x3 matrix where each cell contains a set of three digits. The matrix is:

Row1: [8 6 5] [3 8 0] [4 3 2]
Row2: [3 6 5] [4 8 0] [8 3 2]
Row3: [4 6 5] [8 8 0] [?]

We need to find the missing cell.

Observing the pattern: Each column might have a relationship. Look at the first element 
…(省略中间 thinking)…
the digits in each triple are enclosed in square brackets, so we need to output as [3 3 2]. Also note the format: "Reply with ONLY the missing cell on a single line in the format [d1 d2 ...]. Do not include any explanation."

So final answer: [3 3 2]
```

**[解读]** thinking 步骤里能看到 reasoner 显式枚举/验证规则;chat 在没有这种结构时往往凭直觉给一个 plausible 但错的答案。这是 reasoning vs chat 提升的微观证据。

### Case D: 模型规模分裂 — small 错,大模型对

`item_id = two_rule_comb1__0001`
**Subtype**: `two_rule_comb1`

**Prompt(模型实际看到的)**:
```
You are solving a 3x3 matrix reasoning problem. Each cell of the matrix is a set of digits enclosed in square brackets. The bottom-right cell is missing (marked as [?]). Find the cell that completes the matrix following the same pattern as the other rows/columns.

[8 2] [2 7] [7 9]
[7 2] [8 7] [2 9]
[2 2] [7 7] [?]

Reply with ONLY the missing cell on a single line in the format [d1 d2 ...]. Do not include any explanation.
```

**正确答案**:`[8, 9]`

**各模型响应**:

| Model | Correct? | Parsed | Response (head) |
|---|:-:|---|---|
| gpt-4.1-nano | ❌ | `[2, 9]` | [2 9] |
| gpt-4o-mini | ❌ | `[9, 9]` | [9 9] |
| gpt-4.1-mini | ❌ | `[9, 9]` | [9 9] |
| gpt-4.1 | ✅ | `[8, 9]` | [8 9] |
| gpt-5-mini | ✅ | `[8, 9]` | [8 9] |
| o3-mini | ✅ | `[8, 9]` | [8 9] |

**[解读]** 这类 case 是 Ext2 规模效应曲线的典型驱动 — 题目需要 model 整合两条规则,nano 档没能力,full 档稳过。

### Case E: Logic 类'超人'case — human 39%,top 模型 ≥85%

`item_id = c3_set_union__0006`
**Subtype**: `c3_set_union`

**Prompt(模型实际看到的)**:
```
You are solving a 3x3 matrix reasoning problem. Each cell of the matrix is a set of digits enclosed in square brackets. The bottom-right cell is missing (marked as [?]). Find the cell that completes the matrix following the same pattern as the other rows/columns.

[6] [0] [6 0]
[1] [4] [1 4]
[6 1] [0 4] [?]

Reply with ONLY the missing cell on a single line in the format [d1 d2 ...]. Do not include any explanation.
```

**正确答案**:`[6, 1, 0, 4]`

**各模型响应**:

| Model | Correct? | Parsed | Response (head) |
|---|:-:|---|---|
| deepseek-chat | ✅ | `[6, 1, 0, 4]` | [6 1 0 4] |
| gpt-4o-mini | ❌ | `[6, 4]` | [6 4] |
| gpt-4.1-mini | ❌ | `[6, 4]` | [6 4] |
| deepseek-reasoner | ❌ | `None` | [thinking 截断,无 response_text] |
| gpt-5 | ✅ | `[6, 0, 1, 4]` | [6 0 1 4] |
| o3-mini | ✅ | `[6, 1, 0, 4]` | [6 1 0 4] |
| o4-mini | ✅ | `[6, 1, 0, 4]` | [6 1 0 4] |

**[解读]** logic 类(AND/XOR/set-union 等)是 human 最弱 (39%) 的子类,但 frontier 模型已经在这一类上彻底碾压 — 这是 emergent reasoning 在 4 年内最大的拐点。
