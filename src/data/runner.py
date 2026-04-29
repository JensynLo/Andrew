"""Terraria Data Processing Main Entry"""

import json
import argparse
import os
from typing import Dict, Any

from .interface import TerrariaDataProcessor


def load_data_files(config_path: str) -> Dict[str, Any]:
    """Load all data files according to configuration file

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing all data
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfgs = json.load(f)

    drops_path = cfgs["inpaths"]["drops"]
    items_path = cfgs["inpaths"]["items"]
    npcs_path = cfgs["inpaths"]["npcs"]
    recipes_path = cfgs["inpaths"]["recipes"]

    data_dict = {
        "drops": load_json_file(drops_path),
        "items": load_json_file(items_path),
        "npcs": load_json_file(npcs_path),
        "recipes": load_json_file(recipes_path),
    }

    return data_dict


def load_json_file(file_path: str) -> list:
    """Load JSON file

    Args:
        file_path: Path to the JSON file to load

    Returns:
        Loaded data as a list
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_output_directory(output_dir: str):
    """Ensure output directory exists

    Args:
        output_dir: Path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Build a Terraria knowledge graph from JSON data and generate datasets."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="configs/data_configs.json",
        help="Path to the configuration JSON file containing input paths.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save the generated datasets.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500000,
        help="Number of samples to generate for each dataset type.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.cfg):
        print(f"Error: Config file {args.cfg} does not exist")
        return

    print("Loading data files...")
    try:
        data_dict = load_data_files(args.cfg)
        print(f"Data loading completed, total {len(data_dict)} data types")
    except Exception as e:
        print(f"Error: Problem occurred while loading data: {e}")
        return

    ensure_output_directory(args.output_dir)

    processor = TerrariaDataProcessor()

    print("Building knowledge graph...")
    try:
        graph = processor.process(data_dict)
        print(
            f"Knowledge graph construction completed, nodes: {len(graph.nodes())}, edges: {len(graph.edges())}"
        )
    except Exception as e:
        print(f"Error: Problem occurred while building knowledge graph: {e}")
        return

    print("Generating datasets...")
    try:
        processor.export_dataset(graph, args.output_dir, args.samples)
        print("Dataset generation completed!")
    except Exception as e:
        print(f"Error: Problem occurred while generating datasets: {e}")
        return


if __name__ == "__main__":
    main()
