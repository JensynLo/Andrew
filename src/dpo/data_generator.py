import json
import random
import re
import os
from typing import Dict, List, Tuple, Optional
from jinja2 import Environment, FileSystemLoader
import networkx as nx


class DPODataGenerator:
    """Advanced DPO data pair generator for Terraria assistant"""

    def __init__(self, G: nx.DiGraph | None = None):
        """
        Args:
            G: Terraria Knowledge Graph (DiGraph)
        """
        self.G = G
        if G:
            # Pre-sort nodes by length descending for greedy matching
            self.nodes_sorted = sorted(
                [str(node) for node in G.nodes()], key=len, reverse=True
            )
            # Build a large regex pattern for entity recognition
            # Escaping nodes and using word boundaries
            escaped_nodes = [re.escape(node) for node in self.nodes_sorted]
            self.entity_pattern = re.compile(
                r"\b(" + "|".join(escaped_nodes) + r")\b", re.IGNORECASE
            )

    def extract_instruction_output_pairs(
        self, sft_data: List[Dict]
    ) -> List[Tuple[str, str, Optional[str]]]:
        """Extract instruction-output pairs from SFT data"""
        pairs = []
        for item in sft_data:
            system_prompt = item.get("system", None)
            if "instruction" in item and "output" in item:
                pairs.append((item["instruction"], item["output"], system_prompt))
            elif "messages" in item and len(item["messages"]) >= 2:
                messages = item["messages"]
                for i in range(len(messages)):
                    if messages[i]["role"] == "user":
                        history = ""
                        for j in range(i):
                            role = messages[j]["role"]
                            content = messages[j]["content"]
                            history += f"{role.capitalize()}: {content}\n"
                        instruction = (
                            f"Context:\n{history}\nUser: {messages[i]['content']}"
                            if history
                            else messages[i]["content"]
                        )
                        for j in range(i + 1, len(messages)):
                            if messages[j]["role"] == "assistant":
                                pairs.append(
                                    (instruction, messages[j]["content"], system_prompt)
                                )
                                break
        return pairs

    def generate_rejected_response(self, chosen: str) -> str:
        """Generate a hard negative response"""
        strategies = [
            self._inject_factual_errors,
            self._selective_detail_removal,
            self._introduce_hallucination,
        ]
        
        # Apply 1 or 2 strategies
        num = random.randint(1, 2)
        selected = random.sample(strategies, num)
        
        result = chosen
        for strategy in selected:
            result = strategy(result)
            
        return result

    def _inject_factual_errors(self, text: str) -> str:
        """Use knowledge graph to inject realistic factual errors"""
        if not self.G:
            return text
            
        matches = list(self.entity_pattern.finditer(text))
        if not matches:
            return text
            
        # Select 1-3 random entities to replace
        num_to_replace = min(len(matches), random.randint(1, 3))
        to_replace = random.sample(matches, num_to_replace)
        
        # We need to replace from back to front to maintain indices
        to_replace.sort(key=lambda x: x.start(), reverse=True)
        
        modified_text = text
        for match in to_replace:
            original_name = match.group(0)
            # Find the actual node name in G (case insensitive)
            actual_node = next(
                (n for n in self.G.nodes() if str(n).lower() == original_name.lower()), 
                None
            )
            
            if actual_node:
                node_type = self.G.nodes[actual_node].get("node_type", "Item")
                # Find peers of the same type
                peers = [
                    n for n, d in self.G.nodes(data=True) 
                    if d.get("node_type") == node_type and n != actual_node
                ]
                
                if peers:
                    replacement = random.choice(peers)
                    modified_text = (
                        modified_text[:match.start()] + 
                        replacement + 
                        modified_text[match.end():]
                    )
                    
        return modified_text

    def _selective_detail_removal(self, text: str) -> str:
        """Remove critical structured information sections"""
        # Patterns for common Terraria info blocks
        patterns = [
            r"- \*\*Crafting Recipe\*\*:.*?(?=\n- |\n\n|$)",
            r"- \*\*Drops\*\*:.*?(?=\n- |\n\n|$)",
            r"- \*\*Stats\*\*:.*?(?=\n- |\n\n|$)",
            r"- \*\*How to obtain\*\*:.*?(?=\n- |\n\n|$)",
        ]
        
        modified_text = text
        for pattern in patterns:
            if random.random() > 0.4: # 60% chance to remove a section
                modified_text = re.sub(pattern, "", modified_text, flags=re.DOTALL)
                
        if modified_text == text: # If nothing removed, just truncate
            lines = text.split("\n")
            if len(lines) > 5:
                modified_text = "\n".join(lines[:len(lines)//2])
                
        return modified_text.strip()

    def _introduce_hallucination(self, text: str) -> str:
        """Add incorrect gameplay advice"""
        hallucinations = [
            "Note: This item is only obtainable in Journey Mode.",
            "Warning: Using this item in Hardmode will permanently corrupt your world.",
            "Tip: You can trade this item with the Guide for a Zenith.",
            "This item is dropped by Green Slimes with a 100% drop rate.",
            "You need to be underwater for this item to work correctly.",
        ]
        return text + "\n\n" + random.choice(hallucinations)

    def validate_pair(self, chosen: str, rejected: str) -> bool:
        """Validate the preference pair"""
        if not chosen or not rejected:
            return False
        if chosen.strip() == rejected.strip():
            return False
        # No 0.95 similarity check here - we want hard negatives!
        return True

    def create_dataset(self, sft_paths: List[str], output_path: str, max_samples: int = 5000):
        all_pairs = []
        for path in sft_paths:
            if not os.path.exists(path):
                continue
            with open(path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            pairs = self.extract_instruction_output_pairs(data)
            for inst, chosen, sys in pairs:
                rejected = self.generate_rejected_response(chosen)
                if self.validate_pair(chosen, rejected):
                    all_pairs.append({
                        "instruction": inst,
                        "chosen": chosen,
                        "rejected": rejected,
                        "system": sys if sys else ""
                    })
                    if len(all_pairs) >= max_samples:
                        break
            if len(all_pairs) >= max_samples:
                break
                
        with open(output_path, 'w', encoding='utf-8') as f:
            for p in all_pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"Generated {len(all_pairs)} DPO pairs at {output_path}")
