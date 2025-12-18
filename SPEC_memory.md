## 1. 记忆库概述
### 1.1 什么是 ReasoningBank？
ReasoningBank 是一个**自我进化的智能体框架**

**核心理念**：传统的 LLM Agent 在做任务（例如Alfworld）时往往"做完就忘"，下次遇到类似问题还会重蹈覆辙。ReasoningBank 让 Agent 能够：

1. **从经验中学习**：无论成功还是失败，都能提炼出**可复用的策略**
2. **记忆存储与检索**：将提炼的经验存入记忆库，**遇到新问题时检索相关经验（在alfworld中，根据task goal相似性检索）**
3. **自我进化**：随着经验积累，Agent 的能力持续提升

本项目核心目标是构建一个具备**自我进化能力**的智能体系统，具体包括：

1. **ReasoningBank：** 一个能从成功和失败中提取**可泛化推理策略**的记忆库。
2. **MaTTS (Memory-aware Test-Time Scaling)：** 利用推理时计算（Test-time Compute）来扩大经验规模，与记忆机制形成正向循环。

让LLM Agent 能在多轮交互任务（Alfworld）中实现持续学习。

### 1.2 核心特性
| 特性 | 描述 |
| --- | --- |
| **ReasoningBank** | 推理记忆库，存储结构化的推理策略而非原始轨迹 |
| **MaTTS** | 记忆感知测试时扩展，通过并行扩展获取高质量经验 |
| **双向学习** | 同时从成功和失败中提取经验 |
| **模块化设计** | LLM、环境、记忆、Prompt 解耦，便于扩展 |


### 1.3 与传统方法的区别
| 维度 | 传统 RAG/Memory | ReasoningBank |
| --- | --- | --- |
| **存储内容** | 原始轨迹、文档片段 | 结构化推理策略 |
| **失败处理** | 忽略或丢弃 | 主动提取"避坑指南" |
| **泛化能力** | 基于表面相似度 | 基于推理逻辑相似度 |
| **进化性** | 静态 | 动态进化 |


---

## 2. 核心概念
### 2.1 ReasoningBank：基于推理的记忆库
ReasoningBank 不是简单地存储数据，而是存储经过提炼的**结构化记忆项 (Memory Items)**。

+ **记忆项结构：** 包含三个部分：
    - **标题 (Title)：** 策略的核心标识。
    - **描述 (Description)：** 简短的摘要。
    - **内容 (Content)：** 提炼出的推理步骤、决策理由或操作洞察。
+ **闭环流程：**
    1. **检索 (Retrieval)：** 给定新任务，检索最相关的 $ k $ 个记忆项（用户设置参数，默认Top-K=1，相似度阈值=0.5）。
    2. **执行 (Execution)：** 利用记忆辅助决策，生成轨迹。
    3. **评估：** 任务完成后环境提供的最后结果反馈判断任务是成功还是失败。
    4. **提取 (Extraction)：**
        * **成功轨迹：** 提取有效的操作策略。
        * **失败轨迹：** 提取反事实信号和预防性教训（即“不要做什么”）。
    5. **整合 (Consolidation)：** 将新提取的记忆项存入 ReasoningBank。

ReasoningBank 的核心工作流程——进化循环：

```plain
1. 检索 (Retrieval)     → 根据当前问题，检索相关记忆
2. 执行 (Execution)     → Agent 结合记忆解决问题
3. 评估 (Evaluation)    → 判断任务成功/失败
4. 提取 (Extraction)    → 从轨迹中提取新记忆
5. 整合 (Consolidation) → 将新记忆存入记忆库
```

### 2.4 MaTTS 扩展
### 2.2 记忆项 (Memory Item)
记忆是 ReasoningBank 的核心数据单元：

```python
[
  {
    "memory_id": "memory_unique_id"
    "task_id": "task entry unique_id",
    "task_type": "task type name/id"
    "query": "user query or task goal", // in alfworld, it is task goal
    "trajectory": // Raw trajectory (action-observation list),
    "is_success": True/False,
    "memory_items": [
       // List of Memory Item Schema objects
    ]
  }
]
```

每个记忆（策略）包含1~3个**记忆条目（让LLM提取时自行决定）**：

```json
{
  "title": "Strategy Name (e.g., 'Modular Arithmetic Check')",
  "description": "One-sentence summary of applicability.",
  "content": "Detailed actionable insight on the technique or logic."
}
```

检索到的记忆插入到system_prompt中：

```plain
==================================================
RELEVANT EXPERIENCE FROM SIMILAR TASKS
==================================================
Below are some memory items that accumulated from past interaction from the environment
that may be helpful to solve the task. You can use it when you feel it’s relevant.

[Experience #1]
Goal: [similar goal]
Trajectory: [action sequence]
Correctness: [success/failure]
[Experience #2, #3, ...]
```

### 2.3 **MaTTS：记忆感知测试时扩展（可选的额外功能，默认关闭）**
通过增加计算量获取更高质量的记忆以加速学习，主要采用**重复扩展：**

+ **Parallel Scaling (Best-of-N，用户设置参数，默认N=3):**
    - **针对同一 Query，让Agent 重复该任务实例 N 次 (Temperature稍高，以获取不同轨迹)。**
    - **Contrastive Extraction: **将 $ N $ 条轨迹打包发给 LLM，轨迹可能有的成功有的失败，也有可能全部失败或成功。提示LLM**对比这些轨迹，识别导致成功的一致性模式，或是共同失败的根源，以剔除偶然因素，从而提取更高质量的记忆。 **

| 模式 | 机制 | 优势 |
| --- | --- | --- |
| **重复扩展** | 生成 N 条轨迹，对比成功/失败 | 过滤伪相关，提取一致性模式 |


---

## 3 项目实现具体要求
1. 解耦实现记忆提取、记忆库管理和检索
2. 记忆库管理目前只需要实现记忆检索和添加记忆项功能，后续可能增加Delete，Refine等高级管理操作
3. 用户可以通过配置文件自定义嵌入模型sentence-transformer（默认使用BAAI/bge-base-large-v1.5），检索超参数（top-k，决定是否使用检索到的记忆的相似度阈值），MATTS采样超参数（采样数N，采样温度，max_tokens等）
4. 记忆库保存在本地jsonl文件，并附带嵌入文件以方便下次快速检索而无需反复使用嵌入模型编码，记忆库的命名需要包含任务名称等关键必要信息，相同任务使用相同的记忆库（如alfworld测评就维护一个alfworld上的记忆库）
5. 结果保存时，如果使用了记忆，则任务实例的结果记录中需要保存对应的记忆id，title，检索相似度
6. 整套记忆库的实现要合理、工程上solid，与原来的项目良好整合，有较为鲁棒的错误检测机制，记忆提取出现错误时应当跳过而不是完全中断评测程序。
7. 实现在记忆提取阶段，LLM返回的json数组（记忆项）的较为鲁棒的解析，可以使用json-repair库。
8. 当记忆库已经存在后，下一次测试用户可以通过配置选择：（a,c不会修改记忆库）
    1. 仅评测（baseline）
    2. 边检索边提取记忆
    3. 仅检索不提取经验地评测。
9. 与Baseline测评一样，实现debug模式详细信息输出和断点续传

