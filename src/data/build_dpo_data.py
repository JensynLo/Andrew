import json
import random
import re
import os
from typing import Dict, List, Tuple, Optional
from jinja2 import Environment, FileSystemLoader
import networkx as nx


class DPORetriever:
    """DPO数据构建工具类"""

    def __init__(self, G: nx.DiGraph | None = None, template_dir: str = "template"):
        """
        初始化DPO构建器
        Args:
            G: 包含 Terraria 知识的 DiGraph 实例，用于生成更准确的事实错误
            template_dir: 模板文件存放目录
        """
        self.G = G
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True
        )

    def extract_instruction_output_pairs(
        self, sft_data: List[Dict]
    ) -> List[Tuple[str, str, Optional[str]]]:
        """
        从SFT数据中提取指令-输出对，并保留系统提示词
        Args:
            sft_data: SFT格式的数据列表
        Returns:
            [(instruction, output, system_prompt), ...] 格式的元组列表
        """
        pairs = []

        for item in sft_data:
            system_prompt = item.get("system", None)

            if "instruction" in item and "output" in item:
                # 标准QA格式
                instruction = self.clean_text(item["instruction"])
                output = self.clean_text(item["output"])
                pairs.append((instruction, output, system_prompt))

            elif "messages" in item and len(item["messages"]) >= 2:
                # 多轮对话格式，需要处理上下文
                # 将前面的消息历史加入到指令中，确保指代明确
                messages = item["messages"]

                # 遍历所有用户-助手对
                for i in range(len(messages)):
                    if messages[i]["role"] == "user":
                        # 构造包含历史上下文的指令
                        history_context = ""
                        for j in range(i):
                            role = messages[j]["role"]
                            content = self.clean_text(messages[j]["content"])
                            history_context += f"{role.capitalize()}: {content}\n"

                        user_content = self.clean_text(messages[i]["content"])
                        instruction = (
                            f"Context:\n{history_context}\nUser: {user_content}"
                            if history_context
                            else f"User: {user_content}"
                        )

                        # 查找对应的助手回复
                        for j in range(i + 1, len(messages)):
                            if messages[j]["role"] == "assistant":
                                output = self.clean_text(messages[j]["content"])
                                pairs.append((instruction, output, system_prompt))
                                break  # 只取下一个助手回复

            elif "instruction" in item and "response" in item:
                # 其他格式
                instruction = self.clean_text(item["instruction"])
                output = self.clean_text(item["response"])
                pairs.append((instruction, output, system_prompt))

        return pairs

    def clean_text(self, text: str) -> str:
        """
        清理文本，修复格式问题
        """
        if not text:
            return ""

        # 修复多余的逗号和空值
        text = re.sub(r",\s*,", ", ", text)  # 修复双逗号
        text = re.sub(r",\s+", ", ", text)  # 规范化逗号后的空格

        # 移除多余的连续逗号
        text = re.sub(r",+", ",", text)

        # 修复空内容
        text = re.sub(r",\s+,", ", ", text)
        text = re.sub(r",\s+,", ", ", text)  # 多次运行以防多重逗号

        return text.strip()

    def generate_suboptimal_response(self, good_response: str) -> str:
        """
        生成质量较低的响应作为负样本
        Args:
            good_response: 高质量响应
        Returns:
            质量较低的响应
        """
        # 选择一种或多种降级策略
        strategies = [
            self._remove_context_info,
            self._add_factual_errors,
            self._reduce_detail_level,
            self._introduce_irrelevance,
        ]

        # 随机选择1-2种策略进行组合
        num_strategies = random.randint(1, 2)
        selected_strategies = random.sample(strategies, num_strategies)

        # 依次应用选中的策略
        result = good_response
        for strategy in selected_strategies:
            result = strategy(result)

        return result

    def _remove_context_info(self, response: str) -> str:
        """移除上下文信息"""
        lines = response.split("\n")
        # 移除一些重要信息如警告、进度提示等
        filtered_lines = []
        for line in lines:
            if not any(
                keyword in line.lower()
                for keyword in ["progress", "warning", "⚠️", "threshold"]
            ):
                filtered_lines.append(line)
            else:
                # 保留部分结构但删除关键内容
                if "progress" in line.lower():
                    filtered_lines.append("The item is available in the game.")
                else:
                    filtered_lines.append(line.replace("⚠️", "").strip())

        return "\n".join(filtered_lines).strip()

    def _add_factual_errors(self, response: str) -> str:
        """添加事实性错误，利用知识图谱生成更具迷惑性的错误"""
        response_copy = response

        # 如果有知识图，尝试使用更真实的错误替换
        if self.G is not None:
            # 在响应中查找可能的物品名称，并用相似类型的物品替换
            # 这里我们尝试在响应文本中找到提及的物品名
            words = response.split()
            possible_items = []

            # 寻找可能的物品名称
            for word in words:
                clean_word = re.sub(r"[^\w\s]", "", word)  # 移除标点
                # 尝试匹配节点
                for node in self.G.nodes():
                    if (
                        clean_word.lower() in node.lower()
                        or node.lower() in clean_word.lower()
                    ):
                        possible_items.append(node)
                        break

            # 随机选择一些物品进行替换
            if possible_items:
                for item in random.sample(possible_items, min(2, len(possible_items))):
                    # 找到相似类型的其他物品进行替换
                    item_type = self.G.nodes[item].get("type", "")

                    # 找到同类别的其他物品
                    similar_items = []
                    for node in self.G.nodes():
                        if (
                            node != item
                            and self.G.nodes[node].get("type", "") == item_type
                        ):
                            similar_items.append(node)

                    if similar_items:
                        replacement_item = random.choice(similar_items)
                        response_copy = response_copy.replace(item, replacement_item)

        # 如果没有图或者没有找到合适替换，使用传统的错误注入方法
        if response_copy == response:
            # 使用传统的错误替换，但现在更精确地查找要替换的内容
            # 找到"Recipe"部分并做更精准的替换
            recipe_pattern = r"(- \*\*Crafting Recipe\*\*:.*?)(?=\\n- |\\n$)"
            matches = re.findall(recipe_pattern, response, re.DOTALL)

            if matches:
                for match in matches:
                    # 尝试将配方中的某些材料替换成不合适的材料
                    new_match = match
                    # 简单替换一些常见的材料
                    error_replacements = [
                        ("Iron Bar", "Wooden Bar"),
                        ("Copper Bar", "Silver Coin"),  # 明显错误的材料
                        ("Work Bench", "Stone Block"),
                        ("Sawmill", "Furnace"),
                        ("Demonite Ore", "Dirt Block"),
                        ("Hellstone Bar", "Cobweb"),
                    ]

                    # 随机选择几个进行替换
                    for old_val, new_val in random.sample(
                        error_replacements, k=min(2, len(error_replacements))
                    ):
                        new_match = new_match.replace(old_val, new_val)

                    response_copy = response_copy.replace(match, new_match)

        return response_copy.strip()

    def _reduce_detail_level(self, response: str) -> str:
        """降低详细程度"""
        # 提供一个非常简短的回答
        if random.random() > 0.5:
            simplified_responses = [
                "You can get this item in the game. Look around for crafting recipes or enemy drops.",
                "Check the wiki for this item's acquisition method.",
                "You need to progress further in the game to get this item.",
                "Try defeating some bosses to unlock this item.",
                "Search for crafting recipes that might lead to this item.",
            ]
            return random.choice(simplified_responses)
        else:
            # 只保留部分信息
            lines = response.split("\n")
            # 保留前几行，去掉后面详细的说明
            return "\n".join(lines[: min(5, len(lines))])

    def _introduce_irrelevance(self, response: str) -> str:
        """引入无关信息"""
        irrelevant_parts = [
            "\n\nAdditional tip: Remember to always save your game!",
            "\n\nAlso consider your character's health when farming this item.",
            "\n\nThis item is popular among players who enjoy the end-game content.",
            "\n\nMany players find this item useful for PVP battles.",
            "\n\nThe item has a cool design that matches other items in the game.",
        ]

        # 随机添加一些不相关的信息
        if random.random() > 0.3:
            response += random.choice(irrelevant_parts)

        return response

    def create_dpo_pair(
        self, instruction: str, good_response: str, system_prompt: Optional[str] = None
    ) -> Dict:
        """
        创建DPO训练对
        Args:
            instruction: 用户指令
            good_response: 高质量响应（正样本）
            system_prompt: 系统提示词
        Returns:
            DPO格式的字典
        """
        suboptimal_response = self.generate_suboptimal_response(good_response)

        dpo_pair = {
            "instruction": instruction,
            "chosen": good_response,
            "rejected": suboptimal_response,
        }

        # 如果有系统提示词，则添加
        if system_prompt:
            dpo_pair["system"] = system_prompt

        return dpo_pair

    def validate_preference_pair(self, pair: Dict) -> bool:
        """
        验证偏好对的有效性
        Args:
            pair: DPO格式的字典
        Returns:
            是否有效
        """
        required_keys = ["instruction", "chosen", "rejected"]
        # 检查必需字段存在（instruction等必须存在）
        if not all(key in pair for key in required_keys):
            return False

        # 检查内容是否为空
        if (
            not pair["instruction"].strip()
            or not pair["chosen"].strip()
            or not pair["rejected"].strip()
        ):
            return False

        # 检查chosen和rejected是否相同（这表示转换失败）
        if pair["chosen"] == pair["rejected"]:
            return False

        # 检查长度差异（如果太相似则可能存在问题）
        if len(pair["chosen"]) > 0 and len(pair["rejected"]) > 0:
            similarity_ratio = min(len(pair["chosen"]), len(pair["rejected"])) / max(
                len(pair["chosen"]), len(pair["rejected"])
            )
            if similarity_ratio > 0.95:  # 如果过于相似则可能存在问题
                return False

        return True


def load_sft_data(file_path: str) -> List[Dict]:
    """
    加载SFT数据
    Args:
        file_path: 输入文件路径
    Returns:
        数据列表
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_graph_from_json_files(data_paths: Dict[str, str]) -> nx.DiGraph:
    """
    从JSON文件加载Terraria知识图谱
    Args:
        data_paths: 包含不同数据文件路径的字典，格式为：
                   {"drops": path, "items": path, "npcs": path, "recipes": path}
    Returns:
        构建好的NetworkX有向图
    """
    data_dict = {}

    for data_type, path in data_paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data_dict[data_type] = json.load(f)
        else:
            print(f"Warning: {data_type} file {path} not found, using empty list")
            data_dict[data_type] = []

    # 使用GraphBuilder来构建图
    from .graph_builder import GraphBuilder

    builder = GraphBuilder()
    return builder.build_graph(data_dict)


def build_dpo_dataset(
    input_paths: List[str],
    output_path: str,
    max_samples: Optional[int] = None,
    graph_path: Optional[str] = None,
    data_config_path: Optional[str] = None,
):
    """
    构建完整的DPO数据集
    Args:
        input_paths: 输入文件路径列表
        output_path: 输出文件路径
        max_samples: 最大样本数
        graph_path: 知识图谱文件路径（可选）
        data_config_path: 数据配置文件路径（包含drops, items, npcs, recipes路径）
    """
    # 加载图数据
    G = None

    # 优先使用data_config_path构建图
    if data_config_path and os.path.exists(data_config_path):
        try:
            with open(data_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            # 构建数据路径字典
            data_paths = {}
            for key in ["drops", "items", "npcs", "recipes"]:
                if key in config.get("inpaths", {}):
                    data_paths[key] = config["inpaths"][key]
                else:
                    print(
                        f"Warning: {key} path not found in config, trying default paths"
                    )
                    # 尝试默认路径
                    default_path = f"./output/{key.capitalize()}.json"
                    if os.path.exists(default_path):
                        data_paths[key] = default_path

            if data_paths:
                G = load_graph_from_json_files(data_paths)
                print(
                    f"Knowledge graph loaded successfully with {len(G.nodes())} nodes and {len(G.edges())} edges"
                )
        except Exception as e:
            print(f"Could not load graph from config {data_config_path}, error: {e}")

    # 如果通过配置未成功加载图，再尝试直接使用graph_path
    elif graph_path:
        try:
            import pickle

            with open(graph_path, "rb") as f:
                G = pickle.load(f)
            print(
                f"Knowledge graph loaded from pickle file with {len(G.nodes())} nodes and {len(G.edges())} edges"
            )
        except:
            print(f"Could not load graph from {graph_path}, proceeding without it.")

    # 初始化DPO构建器
    dpo_builder = DPORetriever(G=G)
    all_dpo_pairs = []

    for input_path in input_paths:
        print(f"Processing {input_path}...")
        sft_data = load_sft_data(input_path)

        # 提取指令-输出对（包含系统提示词）
        instruction_output_pairs = dpo_builder.extract_instruction_output_pairs(
            sft_data
        )

        # 为每个对创建DPO样本
        for instruction, output, system_prompt in instruction_output_pairs:
            dpo_pair = dpo_builder.create_dpo_pair(instruction, output, system_prompt)

            # 验证配对
            if dpo_builder.validate_preference_pair(dpo_pair):
                all_dpo_pairs.append(dpo_pair)

                # 限制样本数量
                if max_samples and len(all_dpo_pairs) >= max_samples:
                    break

        if max_samples and len(all_dpo_pairs) >= max_samples:
            break

    # 保存DPO数据集
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_dpo_pairs:
            json.dump(pair, f, ensure_ascii=False)
            f.write("\n")

    print(
        f"DPO dataset built successfully with {len(all_dpo_pairs)} samples at {output_path}"
    )
    return all_dpo_pairs


if __name__ == "__main__":
    # 示例使用
    input_files = [
        "./output/blind_qa.jsonl",
        "./output/how_to_get_qa.jsonl",
        "./output/multiturn_dialogues.jsonl",
    ]
    output_file = "./output/dpo_training_data.jsonl"

    dpo_data = build_dpo_dataset(input_files, output_file, max_samples=5000)
