"""Terraria Data Processor Interface"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import networkx as nx
from .graph_builder import GraphBuilder
from .dataset_generator import DatasetGenerator


class DataProcessorInterface(ABC):
    """Unified interface for data processors"""

    @abstractmethod
    def process(self, data_dict: Dict[str, Any]) -> nx.DiGraph:
        """Process data and return graph

        Args:
            data_dict: Input data dictionary

        Returns:
            Processed graph
        """
        pass

    @abstractmethod
    def export_dataset(self, graph: nx.DiGraph, output_path: str, samples: int = 50000):
        """Export dataset from graph

        Args:
            graph: Input graph to export from
            output_path: Path to save the dataset
            samples: Number of samples to generate
        """
        pass


class TerrariaDataProcessor(DataProcessorInterface):
    """Terraria data processor implementation"""

    def __init__(self):
        self.graph_builder = GraphBuilder()

    def process(self, data_dict: Dict[str, Any]) -> nx.DiGraph:
        """Build Terraria knowledge graph from input data

        Args:
            data_dict: Input data dictionary containing game data

        Returns:
            Built knowledge graph
        """
        return self.graph_builder.build_graph(data_dict)

    def export_dataset(self, graph: nx.DiGraph, output_path: str, samples: int = 50000):
        """Export datasets from the knowledge graph

        Args:
            graph: Knowledge graph to export from
            output_path: Directory path to save the datasets
            samples: Number of samples to generate for each dataset type
        """
        dataset_generator = DatasetGenerator(graph)

        how_to_get_path = f"{output_path}/how_to_get_qa.jsonl"
        multiturn_path = f"{output_path}/multiturn_dialogues.jsonl"
        blind_qa_path = f"{output_path}/blind_qa.jsonl"

        dataset_generator.get_qa(how_to_get_path, samples)
        dataset_generator.get_multiturn(multiturn_path, samples)
        dataset_generator.get_blind_qa(blind_qa_path, samples)

        print("Datasets have been generated to the following files:")
        print(f"  - {how_to_get_path}")
        print(f"  - {multiturn_path}")
        print(f"  - {blind_qa_path}")
