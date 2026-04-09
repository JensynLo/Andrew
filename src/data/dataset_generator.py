"""Terraria Dataset Generator Module"""

import json
import random
from typing import Dict, Optional
import networkx as nx


class DatasetGenerator:
    """Terraria dataset generator"""

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

    def __init__(self, G: nx.DiGraph):
        """Initialize dataset generator

        Args:
            G: Terraria knowledge graph
        """
        self.G = G
        self.items = [n for n in G.nodes() if G.nodes[n].get("node_type") == "Item"]

    def _get_node_tier(self, node_name: str) -> int:
        """Heuristic: Infer item tier based on hardmode tag or drop sources.

        Args:
            node_name: Name of the node to get tier for

        Returns:
            Tier level of the node
        """
        base_tier = 4 if self.G.nodes[node_name].get("hardmode") else 0

        in_edges = self.G.in_edges(node_name, data=True)
        max_tier = base_tier
        for source, _, data in in_edges:
            if data.get("edge_type") == "DROPS_TO":
                tier = self.ENTITY_TIER_MAP.get(source, 0)
                max_tier = max(max_tier, tier)
        return max_tier

    def generate_blind_qa(self, target_item: str) -> Optional[Dict]:
        """Generate a Q&A conversation for 'Blind Queries': User doesn't state progress.

        Args:
            target_item: Target item to generate QA for

        Returns:
            Dictionary containing system prompt, instruction, and output, or None if not possible
        """
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

        think_steps = [
            f"Target: Acquire {target_item}",
            f"Player progress: Unknown (Blind Request)",
            f"Item minimum progress requirement: Tier {item_tier}",
            "Strategy: Provide direct acquisition methods, but emphasize the progression threshold to prevent blind grinding.",
        ]

        blocker = [k for k, v in self.ENTITY_TIER_MAP.items() if v == item_tier]
        blocker_name = blocker[0] if blocker else "a higher-tier boss"

        ans = f"Here is how you can obtain the [{target_item}]:\n"
        if materials:
            ans += (
                f"- **Crafting Recipe**: You need to gather {', '.join(materials)}.\n"
            )
            ans += f"- **Crafting Station**: Crafted at a {stations[0] if stations else 'by hand'}.\n"
        elif drops_from:
            ans += f"- **Drop Source**: Drops from {', '.join(drops_from)}.\n"

        if item_tier > 2:
            ans += f"\n⚠️ **Progression Warning**: This is a later-game item. You need to advance your world progression and at least defeat [{blocker_name}] before you can access these materials or enemies. If you haven't reached this point yet, it's best to focus on your current progression first."

        system_prompt = "You are a Terraria game assistant. When a player does not provide their game progress, you must provide the complete acquisition method and make sure to warn the player about the minimum progression threshold for the item."
        output = f"\n\n{ans}"

        return {"system": system_prompt, "instruction": instruction, "output": output}

    def generate_multiturn_dialogue(self, target_item: str) -> Optional[Dict]:
        """Generate a Multi-turn conversation: AI probes for progress before answering.

        Args:
            target_item: Target item to generate multi-turn dialogue for

        Returns:
            Dictionary containing system prompt and message sequence, or None if not possible
        """
        if not self.G.has_node(target_item):
            return None

        item_tier = self._get_node_tier(target_item)

        if item_tier < 3:
            return None

        in_edges = list(self.G.in_edges(target_item, data=True))
        if not in_edges:
            return None

        player_tier = random.randint(0, item_tier - 1)
        player_state_desc = self.PROGRESS_TIERS[player_tier]
        player_boss = [k for k, v in self.ENTITY_TIER_MAP.items() if v == player_tier]
        player_boss_name = player_boss[0] if player_boss else "Wall of Flesh"

        blocker = [k for k, v in self.ENTITY_TIER_MAP.items() if v == item_tier]
        blocker_name = blocker[0] if blocker else "a late-game boss"

        user_msg_1 = f"I want to get the {target_item}, how do I do that?"

        ai_think_1 = f"Target: {target_item}. Minimum requirement: Defeat {blocker_name}. Player progress unknown. Probing for core threshold first."
        ai_msg_1 = f"\n\n{ai_think_1}\n\nThe {target_item} is a very powerful item. To give you the most accurate advice, have you defeated [{blocker_name}] yet? Or what was the last major boss you defeated?"

        user_msg_2 = (
            f"Haven't beaten them yet, I just recently defeated {player_boss_name}."
        )

        ai_think_2 = f"Confirmed player is at Tier {player_tier}. Target requires Tier {item_tier}. Player is attempting a progression skip. Intercepting and pointing out the next logical step."
        ai_msg_2 = f"\n\n{ai_think_2}\n\nSince you've only just defeated {player_boss_name}, you still have quite a long way to go before you can get the {target_item}!\n\nYou cannot obtain it right now because the required materials or drops are progression-locked. Your primary goal right now should be upgrading your gear and preparing to challenge the upcoming bosses (until you defeat {blocker_name}). Only after that can you start gathering the materials for the {target_item}."

        system_prompt = "You are a Terraria Guide. When encountering queries for complex or late-game items, you should proactively ask the player about their progress and provide tailored advice based on their answer to prevent them from attempting impossible progression skips."

        return {
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_msg_1},
                {"role": "assistant", "content": ai_msg_1},
                {"role": "user", "content": user_msg_2},
                {"role": "assistant", "content": ai_msg_2},
            ],
        }

    def generate_how_to_get_qa(self, target_item: str) -> Optional[Dict]:
        """Generate a Q&A conversation for 'How to get X'.

        Args:
            target_item: Target item to generate QA for

        Returns:
            Dictionary containing system prompt, instruction, and output, or None if not possible
        """
        if not self.G.has_node(target_item):
            return None

        in_edges = list(self.G.in_edges(target_item, data=True))
        if not in_edges:
            return None

        player_tier = random.randint(0, 8)
        player_state_desc = self.PROGRESS_TIERS[player_tier]
        item_tier = self._get_node_tier(target_item)

        materials = []
        stations = []
        drops_from = []
        for src, _, data in in_edges:
            etype = data.get("edge_type")
            if etype == "CRAFTS_INTO":
                materials.append(src)
            elif etype == "REQUIRED_FOR":
                stations.append(src)
            elif etype == "DROPS_TO":
                drops_from.append(src)

        think_steps = [
            f"Target: Acquire {target_item}",
            f"Player current progress: Tier {player_tier} ({player_state_desc})",
            f"Item minimum progress requirement: Tier {item_tier}",
        ]

        if materials:
            station_text = (
                f"Station: {', '.join(stations)}" if stations else "Crafted by hand"
            )
            think_steps.append(
                f"Crafting recipe: Requires {', '.join(materials)}. {station_text}."
            )
        if drops_from:
            think_steps.append(f"Drop source: {', '.join(drops_from)}")

        if player_tier >= item_tier:
            think_steps.append(
                "Judgment: Player progress met. Providing specific instructions."
            )

            ans = f"With your current progress of [{player_state_desc}], you can definitely obtain the {target_item}. "
            if materials:
                ans += f"You can craft it using {', '.join(materials)} "
                ans += f"at a {stations[0]}." if stations else "by hand."
            elif drops_from:
                ans += f"You need to defeat {drops_from[0]} to get it to drop."
        else:
            blocker = [k for k, v in self.ENTITY_TIER_MAP.items() if v == item_tier]
            blocker_name = blocker[0] if blocker else "a higher-tier boss"
            think_steps.append(
                f"Judgment: Player attempting to skip progression. Pointing out prerequisites ({blocker_name})."
            )

            ans = f"Sorry, with your current progress of [{player_state_desc}], you cannot obtain the {target_item} yet. "
            if self.G.nodes[target_item].get("hardmode") and player_tier < 4:
                ans += "This is a Hardmode item. You must first go to the Underworld and defeat the [Wall of Flesh] to initiate Hardmode. "
            else:
                ans += f"You need to advance your game progression and at least defeat [{blocker_name}] to unlock the required materials or drops."

        cot_text = "\n".join(think_steps)
        system_prompt = "You are an advanced Terraria game assistant. You must first use the \n tags to perform logical reasoning, checking if the player's current game progress matches the prerequisite conditions for the target item, and then provide your final answer."
        instruction = (
            f"My current progress is: {player_state_desc}. How can I get {target_item}?"
        )
        output = f"\n\n{cot_text}\n\n{ans}"

        return {"system": system_prompt, "instruction": instruction, "output": output}

    def get_qa(self, output_path: str, samples: int = 5000) -> None:
        """Generate How-To-Get QA dataset and save to file.

        Args:
            output_path: Path to save the dataset
            samples: Number of samples to generate
        """
        all_items = self.items
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

    def get_multiturn(self, output_path: str, samples: int = 5000) -> None:
        """Generate multi-turn dialogue dataset and save to file.

        Args:
            output_path: Path to save the dataset
            samples: Number of samples to generate
        """
        all_items = self.items
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

    def get_blind_qa(self, output_path: str, samples: int = 5000) -> None:
        """Generate blind QA dataset and save to file.

        Args:
            output_path: Path to save the dataset
            samples: Number of samples to generate
        """
        all_items = self.items
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
