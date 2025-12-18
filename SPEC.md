# AlfWorld 评测项目实现说明文档

本文档旨在说明如何实现AlfWorld基准的测试脚本以评估和分析LLM Agent在多轮交互任务上的轨迹结果和性能表现。

## 1. ALFWorld 简介
### 1.1 什么是 ALFWorld？
**ALFWorld**（Aligning Text and Embodied Environments for Interactive Learning）是一个结合了文本游戏和具身AI的交互式学习环境。它基于两个重要项目：

+ **ALFRED**（A Benchmark for Interpreting Grounded Instructions for Everyday Tasks）：一个视觉-语言导航与交互数据集
+ **TextWorld**：微软开发的文本冒险游戏框架

ALFWorld 将 ALFRED 中的 3D 家居环境转换为纯文本交互格式，使得我们可以在不需要视觉渲染的情况下测试 Agent 的规划和推理能力。

### 1.2 为什么用 ALFWorld 测试 LLM？
| 优势 | 说明 |
| --- | --- |
| **多轮交互** | 测试 LLM 在多步骤任务中的规划能力 |
| **状态追踪** | 评估 LLM 对环境状态的理解和记忆 |
| **常识推理** | 任务需要家居常识（如"清洗物品需要水槽"） |
| **纠错能力** | 观察 LLM 在错误后能否调整策略 |
| **指令遵循** | 测试 LLM 对任务目标的理解 |


---

## 2. 环境架构
### 2.1 数据目录结构
```plain
alfworld_data/
├── json_2.1.1/
│   ├── train/              # 训练集
│   ├── valid_seen/         # 验证集（见过的场景）
│   ├── valid_unseen/       # 验证集（未见场景）
│   └── valid_train/        # 训练验证集
├── logic/
    ├── alfred.pddl         # PDDL 领域定义
    └── alfred.twl2         # TextWorld 语法文件
```

### 2.2 游戏文件结构
**每个任务实例**包含以下文件：

```plain
pick_and_place_simple-Book-None-SideTable-329/
└── trial_T20190908_050633_745514/
    ├── game.tw-pddl        # 游戏配置文件（包含 PDDL 和语法）
    ├── initial_state.pddl  # 初始状态定义
    └── traj_data.json      # 任务轨迹数据
```

### 2.3 环境交互流程
```plain
┌─────────────┐     观察 (obs)      ┌─────────────┐
│             │ ──────────────────> │             │
│   ALFWorld  │                     │  LLM Agent  │
│  Environment│ <────────────────── │             │
│             │     动作 (action)   │             │
└─────────────┘                     └─────────────┘
       │                                   │
       │  info['admissible_commands']      │
       │  info['won']                      │
       └───────────────────────────────────┘
```

---

## 3. 任务类型详解
ALFWorld 包含 **6 种任务类型**，覆盖不同的家居场景：

### 3.1 任务类型：6 种
| ID | 任务类型 | 描述 |
| --- | --- | --- |
| 1 | `pick_and_place_simple` | 拿取并放置物品 |
| 2 | `look_at_obj_in_light` | 在灯光下查看物品 |
| 3 | `pick_clean_then_place_in_recep` | 清洁后放置 |
| 4 | `pick_heat_then_place_in_recep` | 加热后放置 |
| 5 | `pick_cool_then_place_in_recep` | 冷却后放置 |
| 6 | `pick_two_obj_and_place` | 拿取两个物品并放置 |


### 3.2 数据集划分
| 划分 | 游戏实例数 | 用途 |
| --- | --- | --- |
| `train` | 3553 | 训练集 |
| `valid_seen` | 140 | 验证集（见过的场景） |
| `valid_train` | 200 | 训练验证集 |
| `valid_unseen` | 134 | 验证集（未见过的场景） |


### 3.3 valid_seen 各任务类型实例数
| 任务类型 | 实例数 |
| --- | --- |
| `pick_and_place_simple` | 35 |
| `pick_clean_then_place_in_recep` | 27 |
| `pick_cool_then_place_in_recep` | 25 |
| `pick_two_obj_and_place` | 24 |
| `pick_heat_then_place_in_recep` | 16 |
| `look_at_obj_in_light` | 13 |
| **总计** | **140** |


### 3.4 实例命名示例
每个实例目录名包含**任务类型-目标物品-容器-场景号**：

```plain
look_at_obj_in_light-AlarmClock-None-DeskLamp-323
pick_clean_then_place_in_recep-Bowl-None-Sink-301
pick_heat_then_place_in_recep-Apple-None-Microwave-205
```

---

### 3.5 pick_and_place_simple（拾取放置）
**目标**：将物品从 A 位置移动到 B 位置

**示例任务**：

```plain
Your task is to: put some book on sidetable.
```

**典型解决步骤**：

1. `look` - 查看周围环境
2. `go to bed 1` - 前往床
3. `take book 1 from bed 1` - 拿起书
4. `go to sidetable 1` - 前往边桌
5. `move book 1 to sidetable 1` - 放下书

---

### 3.6 look_at_obj_in_light（灯下检查）
**目标**：在灯光下检查物品（打开灯，拿着物品）

**示例任务**：

```plain
Your task is to: examine the alarmclock with the desklamp.
```

**典型解决步骤**：

1. `go to desk 1` - 前往书桌
2. `take alarmclock 1 from desk 1` - 拿起闹钟
3. `use desklamp 1` - 打开台灯
4. 任务完成！

---

### 3.7 pick_clean_then_place_in_recep（清洗放置）
**目标**：用水槽清洗物品后放到指定位置

**示例任务**：

```plain
Your task is to: clean some mug and put it in coffeemachine.
```

**典型解决步骤**：

1. `go to countertop 1` - 前往台面
2. `take mug 1 from countertop 1` - 拿起杯子
3. `go to sinkbasin 1` - 前往水槽
4. `clean mug 1 with sinkbasin 1` - 清洗杯子
5. `go to coffeemachine 1` - 前往咖啡机
6. `move mug 1 to coffeemachine 1` - 放下杯子

---

### 3.8 pick_heat_then_place_in_recep（加热放置）
**目标**：用微波炉加热物品后放到指定位置

**示例任务**：

```plain
Your task is to: heat some egg and put it in fridge.
```

**典型解决步骤**：

1. 找到并拿起鸡蛋
2. `go to microwave 1` - 前往微波炉
3. `heat egg 1 with microwave 1` - 加热鸡蛋
4. `go to fridge 1` - 前往冰箱
5. `move egg 1 to fridge 1` - 放入冰箱

---

### 3.9 pick_cool_then_place_in_recep（冷却放置）
**目标**：用冰箱冷却物品后放到指定位置

**示例任务**：

```plain
Your task is to: cool some apple and put it in countertop.
```

**典型解决步骤**：

1. 找到并拿起苹果
2. `go to fridge 1` - 前往冰箱
3. `cool apple 1 with fridge 1` - 冷却苹果
4. `go to countertop 1` - 前往台面
5. `move apple 1 to countertop 1` - 放下苹果

---

### 3.10 pick_two_obj_and_place（双物品放置）
**目标**：将两个相同类型的物品放到指定位置

**示例任务**：

```plain
Your task is to: put two cellphone in drawer.
```

**典型解决步骤**：

1. 找到并拿起第一个手机
2. 放到抽屉
3. 找到并拿起第二个手机
4. 放到抽屉

---

## 4. 交互命令列表
### 4.1 导航命令
| 命令 | 格式 | 示例 | 说明 |
| --- | --- | --- | --- |
| look | `look` | `look` | 查看当前位置周围的物品和可到达的位置 |
| go to | `go to [receptacle]` | `go to dresser 1` | 移动到指定容器/位置 |


### 4.2 物品操作
| 命令 | 格式 | 示例 | 说明 |
| --- | --- | --- | --- |
| take | `take [object] from [receptacle]` | `take apple 1 from fridge 1` | 从容器拿起物品 |
| move | `move [object] to [receptacle]` | `move apple 1 to countertop 1` | 放下物品到容器 |
| inventory | `inventory` | `inventory` | 查看当前携带的物品 |


### 4.3 容器操作
| 命令 | 格式 | 示例 | 说明 |
| --- | --- | --- | --- |
| open | `open [receptacle]` | `open fridge 1` | 打开可开关的容器 |
| close | `close [receptacle]` | `close drawer 1` | 关闭容器 |


### 4.4 物品处理
| 命令 | 格式 | 示例 | 说明 |
| --- | --- | --- | --- |
| heat | `heat [object] with [receptacle]` | `heat potato 1 with microwave 1` | 用微波炉加热 |
| clean | `clean [object] with [receptacle]` | `clean mug 1 with sinkbasin 1` | 用水槽清洗 |
| cool | `cool [object] with [receptacle]` | `cool apple 1 with fridge 1` | 用冰箱冷却 |


### 4.5 其他命令
| 命令 | 格式 | 示例 | 说明 |
| --- | --- | --- | --- |
| use | `use [object]` | `use desklamp 1` | 使用/切换物品状态（如开灯） |
| examine | `examine [object/receptacle]` | `examine apple 1` | 检查物品详情 |


### 4.6 重要规则
> ⚠️ **Agent 每次只能携带一个物品**
>
> ⚠️ **必须先 **`go to`** 某位置才能与那里的物品交互**
>
> ⚠️ **某些容器（如冰箱、抽屉）需要先 **`open`** 才能看到/取出里面的物品**
>

---

## 5. 环境安装
### 5.1 快速安装
```bash
# 1. Clone Repo
git clone https://github.com/alfworld/alfworld.git alfworld
cd alfworld

# 2. 从本地仓库安装 ALFWorld (Text版)
pip install -e .

# 3. 下载 PDDL & Game Files
export ALFWORLD_DATA=<storage_path>
python scripts/alfworld-download
```

### 5.2 验证安装
```bash
python3 -c "
import alfworld
import textworld
print('✅ ALFWorld 安装成功')
"
```

---

## 6. 测试结果输出格式
每条测试实例会生成 JSON 格式的轨迹与结果记录文件：

```json
{
  "model": "qwen/qwen3-8b",
  "timestamp": "2025-12-16T10:30:00",
  "config": {
    "num_games": 5,
    "task_types": [1, 2, 3, 4, 5, 6],
    "seed": 42,
    "use_few_shot": true,
    "max_steps": 30,
    "temperature": 0.3,
    "max_tokens": 8192,
  },
  "summary": {
    "total_games": 5,
    "successes": 3,
    "success_rate": 0.6,
    "avg_steps": 12.4
  },
  "results": [
    {
      "task_type": 2,
      "game_id": "..."
      "success": true,
      "steps": 8,
      "actions": ["look", "go to bed 1", "take book 1 from bed 1", ...],
      "observations": [...],
      "game_file": "/path/to/game.tw-pddl"
    },
    ...
  ]
}
```

---

## 7. 评估指标
### 7.1 主要指标
| 指标 | 说明 | 计算方式 |
| --- | --- | --- |
| **成功率 (Success Rate)** | 成功完成任务的比例 | `成功数 / 总任务数` |
| **平均步数 (Avg Steps)** | 完成任务的平均步数 | `总步数 / 任务数` |
| **成功任务平均步数** | 成功完成任务的平均步数 | 只计算成功的任务 |


## 8. Prompt 设计
### 8.1 ReAct 风格
测试脚本采用 **ReAct **风格的 prompt，要求模型输出：

```plain
THINK: [对当前观察的推理]
ACTION: [执行的动作]
```

### 8.2 Prompt 结构
```plain
┌─────────────────────────────────────┐
│  SYSTEM_PROMPT                      │  <- 介绍环境和可用动作
├─────────────────────────────────────┤
│  FEW_SHOT_EXAMPLES (可选)           │  <- 示例交互
├─────────────────────────────────────┤
│  TASK_PROMPT                        │  <- 当前任务观察
├─────────────────────────────────────┤
│  CONVERSATION_HISTORY               │  <- 之前的交互历史
└─────────────────────────────────────┘
```

### 8.2 Prompt 模板
```plain
==================================================
ENVIRONMENT INSTRUCTIONS
==================================================
[Detailed task environment description and rules]
Example: Go to kitchen, pick up apple, put it in bag

==================================================
EXAMPLE DEMONSTRATIONS
==================================================
[Static few-shot examples]
Example 1: Goal: ... | Action: ... | Observation: ...
Example 2: Goal: ... | Action: ... | Observation: ...

==================================================
YOUR CURRENT TASK
==================================================
Goal: [specific task goal]
Help: type 'check valid actions' if action fails
Help: type 'inventory' to check items

==================================================
RECENT HISTORY
==================================================
Observation: [initial environment state]
Action: [previous action]
Observation: [result of previous action]
Action: [previous action]
Observation: [current state]

==================================================
OUTPUT FORMAT
==================================================
You MUST respond in EXACTLY formats:

Think: <your reasoning>
Free-form explanation of your next step

Action: <exact command>
Must be valid command from ENVIRONMENT INSTRUCTIONS with exact names from RECENT HISTORY
```

注意：

1. 为降低难度，需要编写相关代码来实现"check valid actions"这一新增动作（即环境的info['admissible_commands']），为智能体提供当前状态下所有可执行的动作列表。
2. 为减少上下文混乱，只提供最近的历史信息而不是拼接之前交互的所有完整历史。
3. 需要合理确定 system prompt 和 user prompt 部分。

## 9. 项目实现具体要求
1. 易读与可维护性：项目结构清晰、代码在保证逻辑功能完备的基础上简洁易读，具有良好可维护性和扩展性
2. 解耦与配置化：不同的模块 (LLM, prompt) 应当实现解耦，采用配置化设计，用户只需填写yaml文件即可配置包括但不限于如下重要参数（未指定时提供合理默认值）：
    1. LLM服务配置（支持OpenRouter、vLLM等openai兼容格式）
    2. 测试实例数量
    3. 测试任务类型（可多选）
    4. 测试的split划分（valid_seen/valid_train/valid_unsee）
    5. 是否提供 few-shot 示例
    6. 随机数种子
    7. 结果保存目录
    8. 并行测试数（逐条测试太慢，提高评测效率）
    9. 开启调试模式（结构化打印输入给LLM的prompt和LLM原始回复，以及必要的详细过程追踪信息）
3. LLM服务采用OpenAI兼容格式，需要加入错误检测与重试机制以具备一定的鲁棒性。用户只需提供API_BASE_URL和 API_KEY，并指定必要的推理时参数即可（temperature, max_tokens, max_retries, wait_interval等）
4. 实现并行测试功能（重要！）：必须确保并行不会出现混乱，将原先逐条测试变成N条任务实例并行测试，当使用vLLM部署大模型服务时能显著提高效率。
5. 定时保存运行结果，并实现断点续传功能。
6. 使用日志功能，终端打印信息能清晰追踪实时评估状态。



