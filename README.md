# ALFWorld LLM Agent Evaluation

基于 ALFWorld 环境的 LLM Agent 评测框架，使用 ReAct 风格的 prompt 测试大语言模型在多轮交互家居任务中的表现。

## 快速开始

### 1. 安装依赖

```bash
conda activate icml26  # 或你的环境
pip install -r requirements.txt
```

### 2. 下载 ALFWorld 数据

```bash
export ALFWORLD_DATA=./alfworld_data
python -c "import alfworld; alfworld.download()"
```

### 3. 配置

编辑 `config/default.yaml`，设置你的 API 密钥和模型：

```yaml
llm:
  api_base_url: "https://openrouter.ai/api/v1"
  api_key: "your-api-key"
  model: "qwen/qwen-2.5-72b-instruct"
```

### 4. 运行评测

```bash
python run_eval.py
```

## 命令行参数

```bash
python run_eval.py [OPTIONS]

# 常用参数
--config, -c     配置文件路径
--model, -m      模型名称
--num-games, -n  测试游戏数量
--split, -s      数据集划分 (valid_seen/valid_unseen/valid_train)
--task-types, -t 任务类型 (1-6)
--debug, -d      开启调试模式
--no-few-shot    禁用 few-shot 示例

# 示例
python run_eval.py -n 10 --debug
python run_eval.py --model gpt-4 --split valid_unseen
python run_eval.py -t 1 2 3 -n 20
```

## 配置文件说明

```yaml
llm:
  api_base_url: "https://openrouter.ai/api/v1"
  api_key: ""              # API密钥
  model: ""                # 模型名称
  temperature: 0.3         # 采样温度
  max_tokens: 1024         # 最大token数

test:
  num_games: null          # 测试数量 (null=全部)
  task_types: null         # 任务类型 (null=全部, 或 [1,2,3,4,5,6])
  split: "valid_seen"      # 数据集划分
  seed: 42                 # 随机种子
  max_steps: 30            # 每局最大步数

prompt:
  use_few_shot: true       # 使用few-shot示例
  history_length: 30       # 历史记录长度

runtime:
  save_interval: 1         # 保存间隔
  output_dir: "results"    # 输出目录
  debug: false             # 调试模式
```

## 任务类型

| ID | 任务类型 | 描述 |
|----|----------|------|
| 1 | pick_and_place_simple | 拿取并放置物品 |
| 2 | look_at_obj_in_light | 在灯光下查看物品 |
| 3 | pick_clean_then_place_in_recep | 清洁后放置 |
| 4 | pick_heat_then_place_in_recep | 加热后放置 |
| 5 | pick_cool_then_place_in_recep | 冷却后放置 |
| 6 | pick_two_obj_and_place | 拿取两个物品并放置 |

## 输出格式

评测结果保存为 JSON 格式：

```json
{
  "model": "qwen/qwen-2.5-72b-instruct",
  "summary": {
    "total_games": 10,
    "successes": 7,
    "success_rate": 0.7,
    "avg_steps": 12.5,
    "by_task_type": {...}
  },
  "results": [
    {
      "game_id": "...",
      "goal": "put a book on sidetable",
      "success": true,
      "steps": 8,
      "actions": [...],
      "observations": [...]
    }
  ]
}
```

## 项目结构

```
alfworld-eval/
├── config/
│   └── default.yaml      # 默认配置
├── src/
│   ├── agent.py          # ReAct Agent
│   ├── config.py         # 配置管理
│   ├── environment.py    # ALFWorld环境封装
│   ├── evaluator.py      # 评测器
│   ├── llm_client.py     # LLM客户端
│   ├── logging_utils.py  # 日志工具
│   ├── utils.py          # 数据处理工具
│   └── prompts/
│       ├── system.py     # System Prompt
│       └── few_shot.py   # Few-shot示例
├── run_eval.py           # 入口脚本
└── results/              # 评测结果
```

## 特性

- **断点续传**：相同配置自动恢复进度
- **ReAct Prompt**：Think + Action 格式引导推理
- **check valid actions**：Agent 可查询当前可执行动作
- **彩色输出**：实时显示进度和成功率
- **Debug 模式**：记录完整 prompt/response 交互日志

## 评估指标

| 指标 | 说明 |
|------|------|
| Success Rate | 成功完成任务的比例 |
| Avg Steps | 完成任务的平均步数 |
| Success Avg Steps | 成功任务的平均步数 |

