
import os
import re
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import BartTokenizer, BartForConditionalGeneration, GenerationConfig
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from itertools import combinations
from collections import defaultdict
import random

# Define paths
dataset_path = '/content/drive/MyDrive/1599_summaries(Sheet1).csv'
output_dir = '/content/drive/MyDrive/results'
java_folder_path = '/content/drive/MyDrive/sum_java'

# Load the dataset
df = pd.read_csv(dataset_path, encoding='ISO-8859-1')

class BARTSummarizer:
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def generate_detailed_summary(self, code):
        prompt = (
            "Provide a detailed explanation of the following Java code. "
            "Include the purpose of each class, explain the logic step-by-step, "
            "and describe how the components interact."
        )
        inputs = self.tokenizer.encode(prompt + code, return_tensors='pt', max_length=1024, truncation=True)
        summary_ids = self.model.generate(inputs, generation_config=self.generation_config)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 32)
        self.conv2 = GCNConv(32, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GraphProcessor:
    def __init__(self, gnn_model):
        self.gnn_model = gnn_model

    def networkx_to_pyg_graph(self, nx_graph):
        if len(nx_graph.edges) == 0:
            print("No edges found in the graph.")
            return Data()

        edge_index = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()
        x = torch.eye(len(nx_graph.nodes))  # Node features: identity matrix
        return Data(x=x, edge_index=edge_index)

    def process_graph(self, graph):
        pyg_graph = self.networkx_to_pyg_graph(graph)
        self.gnn_model.eval()
        with torch.no_grad():
            output = self.gnn_model(pyg_graph.x, pyg_graph.edge_index)
        return output

def extract_file_references(java_code):
    references = []
    for line in java_code.splitlines():
        if line.strip().startswith("import"):
            ref = line.split()[-1].replace(";", "")
            references.append(ref)
    return references

def build_dependency_graph(java_files):
    dependency_graph = nx.DiGraph()

    for file_path in java_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            java_code = file.read()
            references = extract_file_references(java_code)
            dependency_graph.add_node(file_path)
            for ref in references:
                for other_file in java_files:
                    if ref in other_file:
                        dependency_graph.add_edge(file_path, other_file)

    print(f"Nodes: {dependency_graph.nodes}")
    print(f"Edges: {dependency_graph.edges}")
    return dependency_graph

class CodeProcessor:
    def __init__(self, graph_processor, summarizer):
        self.graph_processor = graph_processor
        self.summarizer = summarizer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_similarity(self, summary1, summary2):
        embeddings = self.embedding_model.encode([summary1, summary2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity

    def extract_classes(self, code):
        blocks = {}

        # Regular expression to match class definitions
        class_pattern = r'(public|private|protected)?\s*(abstract|final)?\s*class\s+(\w+)(.*?)(?=\n\s*(public|private|protected)?\s*(abstract|final)?\s*class\s+\w+|\Z)'

        for class_match in re.finditer(class_pattern, code, re.DOTALL):
            # Extract access modifier and class name
            access_modifier = class_match.group(1) if class_match.group(1) else "default"  # Default if no modifier
            class_name = class_match.group(3)  # Class name
            class_body = class_match.group(4).strip()  # Class body

            # Store class information in blocks
            blocks[class_name] = {
                'type': 'class',
                'code': f'{access_modifier} class {class_name} {class_body}'
            }

            print(f"Class: {class_name}, Access Modifier: {access_modifier}")

            # Extract methods
            method_pattern = re.compile(r'(public|private|protected)?\s*(static)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{')
            methods = method_pattern.findall(class_body)

            for method in methods:
                method_name = method[2]
                if method_name == class_name:
                    print(f"Method: {method_name} (Constructor)")
                else:
                    print(f"Method: {method_name}")

            # Extract variables/objects
            variable_pattern = re.compile(r'(private|public|protected)?\s*(static)?\s*(\w+)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(=\s*[^;]+)?\s*;')
            variables = variable_pattern.findall(class_body)

            for variable in variables:
                variable_type = variable[2]
                variable_name = variable[3]
                if variable_type in blocks:  # Check if the variable type is a class name
                    print(f"Variable/Object: {variable_name} of type {variable_type} (Inherited class)")
                else:
                    print(f"Variable/Object: {variable_name} of type {variable_type}")

        return blocks


    def build_similarity_graph(self, file_summaries, similarity_threshold=0.65):
        similarity_graph = nx.Graph()  # Using an undirected graph for relationships

        # Check relationships within the same file
        keys = list(file_summaries.keys())
        for key in keys:
            summaries = file_summaries[key]
            summary_keys = list(summaries.keys())
            for i in range(len(summary_keys)):
                for j in range(i + 1, len(summary_keys)):
                    summary1 = summaries[summary_keys[i]]
                    summary2 = summaries[summary_keys[j]]
                    similarity = self.compute_similarity(summary1, summary2)
                    if similarity >= similarity_threshold:
                        similarity_graph.add_edge(summary_keys[i], summary_keys[j], weight=similarity)

        # Check relationships across different files
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                summaries1 = file_summaries[keys[i]]
                summaries2 = file_summaries[keys[j]]
                summary_keys1 = list(summaries1.keys())
                summary_keys2 = list(summaries2.keys())

                for summary_key1 in summary_keys1:
                    for summary_key2 in summary_keys2:
                        similarity = self.compute_similarity(summaries1[summary_key1], summaries2[summary_key2])
                        if similarity >= similarity_threshold:
                            similarity_graph.add_edge(summary_key1, summary_key2, weight=similarity)

        return similarity_graph

    def visualize_graph(self, graph):
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(graph, k=1)  # Increase k for more space between nodes

        # Generate colors for nodes
        colors = [plt.cm.viridis(random.random()) for _ in range(len(graph.nodes()))]
        node_colors = {node: colors[i] for i, node in enumerate(graph.nodes())}

        nx.draw(graph, pos, with_labels=True, node_size=3000, node_color=list(node_colors.values()), font_size=12, font_weight='bold', edge_color='gray', alpha=0.7)
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_color='red', font_size=10)

        plt.title("Class Similarity Graph", fontsize=16)
        plt.axis('off')  # Hide axes
        plt.show()


    def process_files(self, java_files, similarity_threshold=0.5):
        combined_graph = build_dependency_graph(java_files)
        connected_components = list(nx.strongly_connected_components(combined_graph))

        file_summaries = {}
        all_summaries = {}  # To hold all summaries for global comparison
        relationships = []

        for component in connected_components:
            combined_code = ""
            component_files = []

            for file_path in component:
                component_files.append(file_path)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        combined_code += file.read() + "\n\n"
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
                    continue

            blocks = self.extract_classes(combined_code)
            detailed_summaries = {}

            # Generate summaries for each class
            for block_name, block_info in blocks.items():
                try:
                    summary = self.summarizer.generate_detailed_summary(block_info['code'])
                    detailed_summaries[block_name] = summary
                    all_summaries[block_name] = summary  # Store globally
                    print(f"Summary for {block_info['type']} {block_name}:\n{summary}\n")
                except Exception as e:
                    print(f"Error generating summary for {block_info['type']} {block_name}: {e}")

            # Collect relationships within this component
            rels = self.find_relationships(detailed_summaries, similarity_threshold)
            relationships.extend(rels)

            file_summaries[tuple(component_files)] = detailed_summaries

        # Collect relationships across all classes in all components
        all_rels = self.find_relationships(all_summaries, similarity_threshold)
        relationships.extend(all_rels)

        # Grouping blocks based on relationships
        grouped_classes = self.group_classes(relationships)
        similarity_graph = self.build_similarity_graph(file_summaries, similarity_threshold=0.65)

        # Visualize the similarity graph
        self.visualize_graph(similarity_graph)



        # Print grouped classes
        print("Grouped Classes:", grouped_classes)
        return grouped_classes, all_summaries,file_summaries


    def find_relationships(self, summaries, similarity_threshold):
        relationships = []
        summary_keys = list(summaries.keys())

        for i in range(len(summary_keys)):
            for j in range(i + 1, len(summary_keys)):
                summary1 = summaries[summary_keys[i]]
                summary2 = summaries[summary_keys[j]]
                similarity = self.compute_similarity(summary1, summary2)
                if similarity >= similarity_threshold:
                    relationships.append((summary_keys[i], summary_keys[j], similarity))

        return relationships

    def group_classes(self, relationships):
        pairs = defaultdict(list)
        for class1, class2, score in relationships:
            if score > 0.65:  # Filter by score
                pairs[class1].append(class2)
                pairs[class2].append(class1)

        # Generating L3 candidates
        l3_candidates = defaultdict(set)
        for class1, class2 in combinations(pairs.keys(), 2):
            if set(pairs[class1]).intersection(set(pairs[class2])):
                l3_candidates[class1].add(class2)

        # Filter candidates
        filtered_l3_candidates = {k: v for k, v in l3_candidates.items() if v}

        return {
            'L2': pairs,
            'L3': filtered_l3_candidates
        }



# Initialize GNN model
gnn_model = GNNModel(in_channels=2, out_channels=2)
graph_processor = GraphProcessor(gnn_model)

# Load the saved model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained(output_dir)
tokenizer = BartTokenizer.from_pretrained(output_dir)

# Set up generation configuration
generation_config = GenerationConfig.from_pretrained(
    "facebook/bart-large-cnn",
    max_length=250,
    min_length=80,
    num_beams=8,
    length_penalty=2.5,
    repetition_penalty=2.0
)

# Initialize the summarizer
summarizer = BARTSummarizer(bart_model, tokenizer, generation_config)

# Initialize CodeProcessor
code_processor = CodeProcessor(graph_processor, summarizer)
grouped_classes, all_summaries,file_summaries= code_processor.process_files(java_files)

# Specify Java files to process
java_files = [os.path.join(java_folder_path, f) for f in os.listdir(java_folder_path) if f.endswith('.java')]

# Process files
code_processor.process_files(java_files)
