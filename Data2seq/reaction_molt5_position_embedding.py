import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Molecular representation model
class MoleculeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(MoleculeEmbedding, self).__init__()
        model_name = "./model/molt5-base"  # molT5 model path
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512, legacy=False)

        # Freeze original pretrained model parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False  # Freeze pretrained model

        # Add a linear layer to project output to target dimension
        self.linear = nn.Linear(self.model.config.hidden_size, embed_dim)
        self._initialize_weights(init_type='xavier', embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.linear.to(self.device)

    def _initialize_weights(self, init_type='xavier', embed_dim=1024):
        """Weight initialization method."""
        if init_type == 'xavier':
            # Xavier initialization (suitable for linear activation)
            nn.init.xavier_normal_(self.linear.weight, gain=nn.init.calculate_gain('linear'))
            nn.init.zeros_(self.linear.bias)
            
        elif init_type == 'orthogonal':
            # Orthogonal initialization (preserve vector orthogonality)
            nn.init.orthogonal_(self.linear.weight, gain=1.0)
            nn.init.zeros_(self.linear.bias)
            
        elif init_type == 'small_normal':
            # Small normal initialization (avoid large early updates)
            std = 0.02  # Smaller std than default initialization
            nn.init.normal_(self.linear.weight, mean=0.0, std=std)
            nn.init.zeros_(self.linear.bias)

        # Optional: add LayerNorm (recommended)
        self.norm = nn.LayerNorm(embed_dim)  # Add after initialization
    
    def forward(self, smiles_list):
        # Encode multiple SMILES strings with tokenizer
        inputs = self.tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Get model output and extract last hidden state
        with torch.no_grad():  # Disable gradient computation
            encoder_outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Get molecular representations
        mol_embeddings = encoder_outputs.last_hidden_state
        adjusted_embeddings = self.linear(mol_embeddings)

        return adjusted_embeddings, attention_mask


# Reaction learning module with Transformer-based directional features
class ReactionMemoryNetwork(nn.Module):
    def __init__(self, embed_dim, device=None, num_heads=8, num_layers=2):
        super(ReactionMemoryNetwork, self).__init__()
        
        # Device setup
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim

        # Learnable positional encoding (sequence length = 2)
        self.position_emb = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)
        nn.init.normal_(self.position_emb.weight, mean=0.0, std=0.02)  # Small-range initialization

        # Transformer encoder block
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=4*embed_dim,
                dropout=0.1,
                activation='gelu',
                batch_first=True  # More intuitive dimension order
            ),
            num_layers=num_layers
        )
        
        # Output projection layer
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*2),
            nn.GELU(),
            nn.LayerNorm(embed_dim*2),
            nn.Linear(embed_dim*2, embed_dim)
        )
        
        # Initialize module parameters
        self._init_weights()

    def _init_weights(self):
        """Unified initialization method."""
        # Transformer initialization
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Projection layer initialization
        for layer in self.output_proj:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, emb1, emb2):
        """
        Inputs:
        emb1: reactant embedding (batch_size, embed_dim)
        emb2: product embedding (batch_size, embed_dim)

        Output:
        relation_vec: reaction relation vector (batch_size, embed_dim)
        """
        # Build sequence tensor: (batch_size, seq_len=2, embed_dim)
        batch_size = emb1.size(0)
        combined = torch.stack([emb1, emb2], dim=1)  # Dimension order: [batch, seq, features]

        # Add positional encoding
        positions = torch.arange(2, device=self.device).expand(batch_size, 2)
        pos_emb = self.position_emb(positions)
        combined = combined + pos_emb

        # Transformer encoding
        transformer_out = self.transformer_encoder(combined)  # (batch, 2, embed_dim)
        
        # Use pooled output over sequence positions
        # Alternative: use the last position only
        relation_vec = transformer_out.mean(dim=1)

        
        # Project into target representation space
        return self.output_proj(relation_vec)

class ReactionRepresentation(nn.Module):
    def __init__(self, embed_dim=1024, device='cuda'):
        super(ReactionRepresentation, self).__init__()
        self.molecule_embedder = MoleculeEmbedding(embed_dim=embed_dim).to(device)
        self.reaction_network = ReactionMemoryNetwork(embed_dim=embed_dim).to(device)
        self.device = device
        self.molecule_embedder.to(self.device)
        self.reaction_network.to(self.device)

    def forward(self, reaction_smiles):
        """
        Process current batch reaction data to build reaction embeddings.
        """
        # Store reactant/product lists
        all_reactants = []
        all_products = []
        reactant_counts = []
        product_counts = []
        
        # Iterate each reaction and collect reactants/products
        for reactant_smiles, product_smiles in reaction_smiles:
            all_reactants.extend(reactant_smiles)
            all_products.extend(product_smiles)
            reactant_counts.append(len(reactant_smiles))
            product_counts.append(len(product_smiles))

        # print(all_reactants)
        # print(all_products)
        # Get reactant and product embeddings
        reactant_embeddings, reactant_mask = self.molecule_embedder(all_reactants)
        product_embeddings, product_mask = self.molecule_embedder(all_products)
        # print(reactant_embeddings.shape)
        # print(product_embeddings.shape)
        # print(reactant_embeddings)
        # print(product_embeddings)

        for i, embedding in enumerate(reactant_embeddings):
            if torch.any(torch.isnan(embedding)) or torch.any(torch.isinf(embedding)):
                print(f"!!!Warning: NaN or Inf detected in reactant {all_reactants[i]} (index {i})")

        # Pool reactant and product embeddings
        reactant_avg_embeddings = torch.sum(reactant_embeddings * reactant_mask.unsqueeze(-1), dim=1) / torch.sum(reactant_mask, dim=1, keepdim=True)
        product_avg_embeddings = torch.sum(product_embeddings * product_mask.unsqueeze(-1), dim=1) / torch.sum(product_mask, dim=1, keepdim=True)

        # Process reactant embeddings
        reaction_embeddings_reactants = self._process_embeddings(reactant_avg_embeddings, reactant_counts)

        # Process product embeddings
        reaction_embeddings_products = self._process_embeddings(product_avg_embeddings, product_counts)

        # Feed pooled embeddings into reaction network
        reaction_embedding = self.reaction_network(reaction_embeddings_reactants, reaction_embeddings_products)

        return reaction_embedding

    def _process_embeddings(self, avg_embeddings, counts):
        """
        Process reactant/product embeddings by splitting and aggregating with per-reaction counts.
        """
        embeddings_reaction_level = []
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            embedding = avg_embeddings[start_idx:end_idx]
            embeddings_reaction_level.append(torch.sum(embedding, dim=0, keepdim=True))  # Aggregate embedding to reaction level
            start_idx = end_idx

        return torch.cat(embeddings_reaction_level, dim=0)
########################################################################






