import json
import random
from typing import Dict, Optional, List
import networkx as nx
from jinja2 import Environment, FileSystemLoader


class DatasetGenerator:
    """泰拉瑞亚数据集生成器核心类"""

    # 玩家游戏进度分级
    PROGRESS_TIERS = {
        0: "Just built first house (Pre-Boss)",
        1: "Defeated Eye of Cthulhu / King Slime",
        2: "Defeated Eater of Worlds / Brain of Cthulhu",
        3: "Defeated Skeletron",
        4: "Defeated Wall of Flesh (Entered Hardmode)",
        5: "Defeated the Mechanical Bosses",
        6: "Defeated Plantera",
        7: "Defeated Golem",
        8: "Defeated Lunatic Cultist",
    }

    # 实体（Boss）对应的进度层级映射
    ENTITY_TIER_MAP = {
        "Eye of Cthulhu": 1,
        "King Slime": 1,
        "Eater of Worlds": 2,
        "Brain of Cthulhu": 2,
        "Skeletron": 3,
        "Wall of Flesh": 4,
        "The Twins": 5,
        "The Destroyer": 5,
        "Skeletron Prime": 5,
        "Plantera": 6,
        "Golem": 7,
        "Lunatic Cultist": 8,
        "Moon Lord": 8,
    }

    def __init__(self, G: nx.DiGraph, template_dir: str = "template"):
        """
        初始化生成器
        Args:
            G: 包含 Terraria 知识的 DiGraph 实例
            template_dir: 模板文件存放目录
        """
        self.G = G
        self.items = [n for n in G.nodes() if G.nodes[n].get("node_type") == "Item"]

        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
        )

        # 预加载模板
        self.system_prompt_template = self.jinja_env.get_template("system_prompt.j2")
        self.how_to_get_template = self.jinja_env.get_template("how_to_get_output.j2")
        self.multiturn_template = self.jinja_env.get_template("multiturn_dialogue.j2")
        self.blind_qa_template = self.jinja_env.get_template("blind_qa_output.j2")

    def _get_node_tier(self, node_name: str) -> int:
        """推断物品的最低进度等级"""
        base_tier = 4 if self.G.nodes[node_name].get("hardmode") else 0
        in_edges = self.G.in_edges(node_name, data=True)
        max_tier = base_tier
        for source, _, data in in_edges:
            edge_type = data.get("edge_type")
            if edge_type == "DROPS_TO":
                tier = self.ENTITY_TIER_MAP.get(source, 0)
                max_tier = max(max_tier, tier)
        return max_tier

    def get_qa(self, output_path: str, samples: int = 5000) -> None:
        """Generate How-To-Get QA dataset and save to file.
        批量生成并保存 How-To-Get (条件获取) 问答数据集

        Args:
            output_path: 保存数据集的文件路径
            samples: 生成的样本数量上限
        """
        all_items = self.items[:]
        random.shuffle(all_items)
        selected_items = all_items[:samples]
        dataset = []
        for item in selected_items:
            qa = self.generate_how_to_get_qa(item)
            if qa:
                dataset.append(qa)

        with open(output_path, "w", encoding="utf-8") as f:
            for data in dataset:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ 成功生成并保存 {len(dataset)} 条 How-To-Get 数据至 {output_path}")

    def get_multiturn(self, output_path: str, samples: int = 5000) -> None:
        """Generate multi-turn dialogue dataset and save to file.
        批量生成并保存多轮对话数据集

        Args:
            output_path: 保存数据集的文件路径
            samples: 生成的样本数量上限
        """
        all_items = self.items[:]
        random.shuffle(all_items)
        selected_items = all_items[:samples]
        dataset = []
        for item in selected_items:
            dialogue = self.generate_multiturn_dialogue(item)
            if dialogue:
                dataset.append(dialogue)

        with open(output_path, "w", encoding="utf-8") as f:
            for data in dataset:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ 成功生成并保存 {len(dataset)} 条 多轮对话 数据至 {output_path}")

    def get_blind_qa(self, output_path: str, samples: int = 5000) -> None:
        """Generate blind QA dataset and save to file.
        批量生成并保存盲查问答数据集

        Args:
            output_path: 保存数据集的文件路径
            samples: 生成的样本数量上限
        """
        all_items = self.items[:]
        random.shuffle(all_items)
        selected_items = all_items[:samples]
        dataset = []
        for item in selected_items:
            qa = self.generate_blind_qa(item)
            if qa:
                dataset.append(qa)

        with open(output_path, "w", encoding="utf-8") as f:
            for data in dataset:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")
        print(f"✅ 成功生成并保存 {len(dataset)} 条 盲查问答 数据至 {output_path}")

    def generate_blind_qa(self, target_item: str) -> Optional[Dict]:
        """生成盲查 QA"""
        if not self.G.has_node(target_item):
            return None
        in_edges = list(self.G.in_edges(target_item, data=True))
        if not in_edges:
            return None

        item_tier = self._get_node_tier(target_item)
        materials = [
            src for src, _, data in in_edges if data.get("edge_type") == "CRAFTS_INTO"
        ]
        stations = [
            src for src, _, data in in_edges if data.get("edge_type") == "REQUIRED_FOR"
        ]
        drops_from = [
            src for src, _, data in in_edges if data.get("edge_type") == "DROPS_TO"
        ]

        questions = [
            f"How do I get {target_item}?",
            f"Can someone tell me where to find {target_item}?",
            f"What's the recipe for {target_item}?",
            f"Where does {target_item} drop?",
        ]
        instruction = random.choice(questions)
        blocker = [k for k, v in self.ENTITY_TIER_MAP.items() if v == item_tier]
        blocker_name = blocker[0] if blocker else "a higher-tier boss"

        context = {
            "target_item": target_item,
            "item_tier": item_tier,
            "materials": materials,
            "stations": stations,
            "drops_from": drops_from,
            "blocker_name": blocker_name,
        }

        return {
            "system": self.system_prompt_template.render(task="blind_qa"),
            "instruction": instruction,
            "output": self.blind_qa_template.render(**context),
        }

    def generate_multiturn_dialogue(self, target_item: str) -> Optional[Dict]:
        """生成多轮对话"""
        if not self.G.has_node(target_item):
            return None
        item_tier = self._get_node_tier(target_item)
        if item_tier < 3:
            return None

        in_edges = list(self.G.in_edges(target_item, data=True))
        if not in_edges:
            return None

        player_tier = random.randint(0, item_tier - 1)
        player_boss = [k for k, v in self.ENTITY_TIER_MAP.items() if v == player_tier]
        player_boss_name = player_boss[0] if player_boss else "Wall of Flesh"
        blocker = [k for k, v in self.ENTITY_TIER_MAP.items() if v == item_tier]
        blocker_name = blocker[0] if blocker else "a late-game boss"

        context = {
            "target_item": target_item,
            "item_tier": item_tier,
            "player_tier": player_tier,
            "player_boss_name": player_boss_name,
            "blocker_name": blocker_name,
        }

        return {
            "system": self.system_prompt_template.render(task="multiturn"),
            "messages": [
                {
                    "role": "user",
                    "content": f"I want to get the {target_item}, how do I do that?",
                },
                {
                    "role": "assistant",
                    "content": self.multiturn_template.render(**context, turn=1),
                },
                {
                    "role": "user",
                    "content": f"Haven't beaten them yet, I just recently defeated {player_boss_name}.",
                },
                {
                    "role": "assistant",
                    "content": self.multiturn_template.render(**context, turn=2),
                },
            ],
        }

    def generate_how_to_get_qa(self, target_item: str) -> Optional[Dict]:
        """生成带进度的获取 QA"""
        if not self.G.has_node(target_item):
            return None
        in_edges = list(self.G.in_edges(target_item, data=True))
        if not in_edges:
            return None

        player_tier = random.randint(0, 8)
        player_state_desc = self.PROGRESS_TIERS[player_tier]
        item_tier = self._get_node_tier(target_item)

        materials, stations, drops_from = [], [], []
        for src, _, data in in_edges:
            etype = data.get("edge_type")
            if etype == "CRAFTS_INTO":
                materials.append(src)
            elif etype == "REQUIRED_FOR":
                stations.append(src)
            elif etype == "DROPS_TO":
                drops_from.append(src)

        blocker = [k for k, v in self.ENTITY_TIER_MAP.items() if v == item_tier]
        blocker_name = blocker[0] if blocker else "a higher-tier boss"

        context = {
            "target_item": target_item,
            "player_tier": player_tier,
            "player_state_desc": player_state_desc,
            "item_tier": item_tier,
            "materials": materials,
            "stations": stations,
            "drops_from": drops_from,
            "blocker_name": blocker_name,
            "is_hardmode": bool(self.G.nodes[target_item].get("hardmode")),
        }

        return {
            "system": self.system_prompt_template.render(task="how_to_get"),
            "instruction": f"My current progress is: {player_state_desc}. How can I get {target_item}?",
            "output": self.how_to_get_template.render(**context),
        }

    def _save_jsonl(self, dataset: List[Dict], path: str):
        with open(path, "w", encoding="utf-8") as f:
            for data in dataset:
                json.dump(data, f, ensure_ascii=False)
                f.write("\n")

    def save_all_datasets(self, samples_per_type: int = 5000):
        """生成并保存所有数据集"""
        all_items = self.items[:]
        random.shuffle(all_items)

        # QA 数据集
        qa_data = [
            self.generate_how_to_get_qa(it) for it in all_items[:samples_per_type]
        ]
        self._save_jsonl([d for d in qa_data if d], "dataset_how_to_get.jsonl")

        # 盲查数据集
        blind_data = [self.generate_blind_qa(it) for it in all_items[:samples_per_type]]
        self._save_jsonl([d for d in blind_data if d], "dataset_blind_qa.jsonl")

        # 多轮对话
        multi_data = [
            self.generate_multiturn_dialogue(it) for it in all_items[:samples_per_type]
        ]
        self._save_jsonl([d for d in multi_data if d], "dataset_multiturn.jsonl")
