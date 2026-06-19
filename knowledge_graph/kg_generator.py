import networkx as nx
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from loguru import logger

class SafetyKnowledgeGraph:
    """
    Knowledge Graph Generator for NariRaksha.
    Maps relationships between Risks, Laws (BNS), Actions, and Scenarios.
    """
    def __init__(self):
        self.graph = nx.Graph()
        
    def build_from_dataset(self, dataset_path: str):
        logger.info(f"Building knowledge graph from {dataset_path}")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            
        for item in data:
            risk = item['risk_type']
            severity = item['severity']
            law = item.get('legal_context', 'General Guidelines')
            
            # Add nodes
            self.graph.add_node(risk, type="Risk")
            self.graph.add_node(severity, type="Severity")
            self.graph.add_node(law, type="Law")
            
            # Add edges
            self.graph.add_edge(risk, severity, relation="has_severity")
            self.graph.add_edge(risk, law, relation="covered_by")
            
        logger.info(f"Built KG with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
    def export_graph(self, output_path: str):
        data = nx.node_link_data(self.graph)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported Knowledge Graph to {output_path}")

if __name__ == "__main__":
    kg = SafetyKnowledgeGraph()
    # kg.build_from_dataset("datasets/synthetic/nariraksha_synthetic.jsonl")
    # kg.export_graph("knowledge_graph/nari_kg.json")
