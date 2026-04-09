# 🌳 Project Andrew: The Intelligent Terraria Assistant

![Terraria](https://img.shields.io/badge/Game-Terraria-green)
![LLM](https://img.shields.io/badge/LLM-Llama--3.2--3B-blue)
![Fine-Tuning](https://img.shields.io/badge/FineTuning-QLoRA-orange)
![Python](https://img.shields.io/badge/Python-3.10%+-blue)

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

