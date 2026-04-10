# 🌳 Andrew: The Intelligent Terraria Assistant

![Terraria](https://img.shields.io/badge/Game-Terraria-green)
![LLM](https://img.shields.io/badge/LLM-Llama--3.2--3B-blue)
![Fine-Tuning](https://img.shields.io/badge/FineTuning-QLoRA-orange)
![Python](https://img.shields.io/badge/Python-3.10+-blue)

**Andrew** 是一个专为[《泰拉瑞亚（Terraria）》](https://store.steampowered.com/app/105600/Terraria/)打造的智能大语言模型（LLM）助手。它不仅仅是一个简单的“Wiki Tutor”，而是一个真正理解游戏**进程逻辑**、**合成依赖树** 和**状态机** 的 AI 向导。

### 💡 为什么叫 "Andrew"？
> 在泰拉瑞亚中，**向导（The Guide）** 是玩家降生于世界后遇到的第一个 NPC，他为迷茫的新手提供生存建议和物品合成指南。而 **Andrew** 不仅是向导最经典的随机名字之一，更是泰拉瑞亚原作者兼首席开发者 Andrew "Redigit" Spinks 的名字。
> 
> 本项目沿用此名，正是为了契合其设计初衷：**成为每一个泰拉玩家身边最可靠、最懂行的专属数字向导。**

---

## ✨ 核心特性 (Key Features)

与通用大模型相比，Project Andrew 解决了垂直领域游戏问答中最致命的“越级胡说”问题：

* **🛡️ 进程感知与防越级 (Progression-Aware)**：内置游戏进程状态机（从肉前到击败月总的 9 大阶段）。当新手玩家询问大后期物品时，Andrew 不会盲目给出合成表，而是会实施“越级拦截”，指出当前需要推进的前置 Boss。
* **🧠 图谱驱动的思维链 (Graph-Driven CoT)**：基于官方 Wiki 爬取构建了包含数千个节点（物品、NPC、制作站）的有向无环图（DAG）。模型的微调数据均通过图遍历算法严密推导生成，确保底层逻辑准确。
* **💬 主动探针与多轮交互 (Multi-turn Probing)**：面对语境缺失的“盲问”（如：直接问“天顶剑怎么合”），Andrew 能够主动反问玩家当前的游戏进度，从而给出最符合玩家现状的定制化建议。

---

## 🚀 快速开始 (Quick Start)

```bash
# 1. 克隆仓库
git clone git@github.com:JensynLo/Andrew.git
cd Andrew

# 2. 安装依赖
pip install -r requirements.txt

# 3. 爬取数据
python -m src.spider.runner --cfg configs/spider_config.json

# 4. 生成微调数据
python -m src.data.runner --cfg configs/data_config.json

# 5. 微调模型
python -m src.main.train

# 6. 对话
python -m src.main.chat
```

---

## 🧩 项目原理与技术架构 (Architecture & Methodology)

Andrew 并不是通过简单地把整个 Wiki 文本塞进大模型来训练的。为了让模型拥有严密的“逻辑推导”能力，本项目采用了一套**“图谱提取 ➔ 逻辑仿真 ➔ 思维链生成”**的完整管线。

### 1. 数据采集与知识图谱构建 (Knowledge Graph)
* **全量图谱**：通过异步爬虫调用官方 Wiki.gg 的 Cargo API，获取了 `Items`（物品）、`Recipes`（配方）、`NPCs`（实体）和 `Drops`（掉落）四大核心数据表。
* **构建 DAG**：利用 `NetworkX` 将这些扁平的数据表转化为一张庞大的有向图（Directed Graph）。节点代表物品或实体，边代表“合成需要 (`CRAFTS_INTO`)”、“掉落自 (`DROPS_FROM`)” 或 “制作站限制 (`REQUIRED_FOR`)”。
* **状态机映射**：手动为核心 Boss 和阶段打上 Tier（例如：肉前为 Tier 0-3，机械三王为 Tier 5）。

### 2. 仿真对话与思维链 (Chain-of-Thought) 生成
在拥有了图谱和状态机后，通过 Python 脚本在图谱上随机游走，生成高质量的指令微调数据集（JSONL）：
* **正向指导**：当虚拟玩家进度 ≥ 物品所需进度时，生成标准的获取路径。
* **越级惩罚 (Progression Skip)**：当虚拟玩家进度不足时，强制模型在 `<think>` 标签中推导前置节点，并在输出中拒绝提供后期配方，转而提示玩家先去击败对应的 Boss。
* **盲问反问 (Blind Probing)**：对于没有提供上下文的高级物品提问，生成多轮对话数据，训练模型主动反问玩家当前的进度。

---

## 🛠️ 大模型底座与微调细节 (Model & Fine-Tuning)

### 1. 底座模型选择
本项目选用 [**`meta-llama/Llama-3.2-3B-Instruct`**](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) 作为底座模型。
* **轻量且强大**：3B 参数规模意味着它可以在 24G 显存的消费级显卡（如 RTX 3090/4090）上轻松完成微调和本地部署。
* **指令遵循**：Llama 3.2 具备极强的 System Prompt 遵循能力和多轮对话理解能力，非常契合我们带有 `<think>` 逻辑标签的复杂任务。

### 2. QLoRA 高效微调 (Efficient Fine-Tuning)
由于直接全量微调大模型成本极高，本项目采用了参数高效微调（PEFT）技术：
* **4-bit 量化 (BitsAndBytes)**：以 NF4 精度加载基座模型，配合双重去量化，大幅降低显存占用。
* **LoRA 适配器**：设置秩 `r=16, alpha=32`，并覆盖了所有线性层 (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) 以最大化 3B 模型的学习表达能力。
* **优化策略**：使用 PyTorch 原生 `adamw_torch` 优化器，配合 `bfloat16` 混合精度训练、梯度累加（Gradient Accumulation）和梯度检查点（Gradient Checkpointing），确保在小显存下安全运行。

### 3. 数据集均衡与混合训练 (Mix Training)
为了防止模型在学习泰拉瑞亚专属逻辑时发生“灾难性遗忘”（即忘记了通用常识或剧情背景），我们在最终微调阶段采用了**混合数据策略**：
1.  **自建逻辑数据**：基于图谱生成的带有 `<think>` 标签的 `how_to_get_qa`、`blind_qa` 和 `multiturn_dialogues`。
2.  **Wiki 常识数据**：从 Hugging Face 引入 `lparkourer10/terraria-wiki` 开源数据集。
3.  **动态下采样 (Downsampling)**：按照固定比例（如 1:1 或 1.5:1）对庞大的 Wiki 数据进行随机采样下压，将其与自建的强逻辑数据混合打乱后再训练。这使得 Andrew 既是一个严谨的“合成逻辑大师”，又是一个博学的“Wiki 百科全书”。

---

## 📂 目录结构 (Directory Structure)

```text
Andrew/
├── configs/               # 爬虫与数据生成的配置文件
├── output/                # 生成的原始 JSON、微调 JSONL 数据集及模型权重输出
├── src/
│   ├── spider/            # Wiki.gg Cargo API 异步爬虫与数据清洗
│   ├── data/              # NetworkX 图谱构建、思维链对话生成器
│   └── main/              # Llama-3 微调脚本 (SFTTrainer) 与终端交互推理代码
├── requirements.txt       # 依赖清单
└── README.md              # 项目文档
```
