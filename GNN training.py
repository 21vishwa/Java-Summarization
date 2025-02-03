import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# GCN model definition
class GCNModel(nn.Module):
    def __init__(self, in_channels, num_nodes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, num_nodes * num_nodes)  # Output size should be num_nodes^2 for adjacency matrix prediction
        self.num_nodes = num_nodes

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)

        # Reshape the output to form a square matrix
        adj_pred = x.view(-1, self.num_nodes, self.num_nodes)
        return adj_pred

# Function to prepare node features and adjacency matrix for training
def preprocess_data(node_features, adjacency_matrix):
    # Extract feature vectors from node features
    feature_list = []
    for class_name, features in node_features.items():
        # Concatenate relevant features into a single vector (e.g., class embedding, number of methods, etc.)
        class_embedding = features['class_embedding']  # (384,)
        num_methods = features['num_methods']  # Scalar
        has_constructor = features['has_constructor']  # Scalar
        num_code_lines = features['num_code_lines']  # Scalar
        summary_embedding = features['summary_embedding']  # (384,)

        # Combine all features into a single tensor
        combined_features = torch.cat([
            torch.tensor(class_embedding),
            torch.tensor([num_methods, has_constructor, num_code_lines], dtype=torch.float32),
            torch.tensor(summary_embedding)
        ])
        feature_list.append(combined_features)

    # Convert to tensor
    node_feature_tensor = torch.stack(feature_list)  # Shape: [num_nodes, feature_dim]

    # Convert adjacency matrix to PyTorch tensor
    adjacency_matrix_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)

    return node_feature_tensor, adjacency_matrix_tensor

# Custom loss function for adjacency matrix prediction
def adjacency_loss(predicted_adj, true_adj):
    return nn.MSELoss()(predicted_adj, true_adj)

# Training function
def train_gcn(node_features, adjacency_matrix, num_epochs=100, learning_rate=0.001):
    # Preprocess the data
    x, true_adjacency_matrix = preprocess_data(node_features, adjacency_matrix)
    num_nodes = true_adjacency_matrix.shape[0]

    # Define GCN model
    gnn_model = GCNModel(in_channels=x.shape[1], num_nodes=num_nodes)

    # Define optimizer
    optimizer = optim.Adam(gnn_model.parameters(), lr=learning_rate)

    # Define edge_index (self-connections as GCN needs edge indices)
    edge_index = torch.arange(x.size(0)).unsqueeze(0).repeat(2, 1)  # Identity edges (self-loops)

    # Training loop
    for epoch in range(num_epochs):
        gnn_model.train()
        optimizer.zero_grad()

        # Forward pass: Predict the adjacency matrix
        predicted_adjacency_matrix = gnn_model(x, edge_index)

        # Compute loss
        loss = adjacency_loss(predicted_adjacency_matrix, true_adjacency_matrix)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) :
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    print("Training complete!")
    return gnn_model

# Example usage:
gnn_model = GCNModel(in_channels=x.shape[1], num_nodes=num_nodes)
gnn_model.load_state_dict(torch.load('/content/drive/MyDrive/gcn_model_weights.pth'))
gnn_model.eval()
# node_features and adjacency_matrix are assumed to be available from your dataset
gnn_model = train_gcn(node_features, adjacency_matrix, num_epochs=200)
torch.save(gnn_model.state_dict(), '/content/drive/MyDrive/gcn_model_weights.pth')
print("Model weights saved!")
