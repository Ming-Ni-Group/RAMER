import torch
import torch.nn as nn
import torch.nn.init as init

class StructureEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(StructureEmbedding, self).__init__()
        # Define a linear layer: input 3072 -> output embed_dim
        self.fc = nn.Linear(3072, embed_dim)
        
        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(3072)
        
        # Initialize parameters with Xavier
        init.xavier_uniform_(self.fc.weight)  # Xavier uniform initialization
        init.zeros_(self.fc.bias)  # Zero bias initialization

    def forward(self, gearnet_embedding_batch):
        # Apply batch normalization first
        output = self.batch_norm(gearnet_embedding_batch)
        
        # Then apply linear projection
        output = self.fc(output)  # (batch_size, embed_dim)
        
        return output
