import torch
import torch.nn as nn
import sys
sys.path.append("./Data2seq")
from structure_embedding import StructureEmbedding # Structure
from reaction_molt5_position_embedding import ReactionRepresentation
from proteinT5_embedding import ProteinSequenceEmbedding # ProteinT5-based protein sequence representation

class BioData2Seq(nn.Module):
    def __init__(self, modality, embed_dim, protein_stage="inference"):
        super(BioData2Seq, self).__init__()
        self.modality = modality
        self.embed_dim = embed_dim
        self.protein_stage = protein_stage
        
        # Initialize embedder for each modality
        if self.modality == 'protein-sequence':
            self.protein_embed = ProteinSequenceEmbedding(
                embed_dim=self.embed_dim,
                stage=self.protein_stage
            )
        elif self.modality == 'structure':
            self.structure_embed = StructureEmbedding(embed_dim=self.embed_dim)
        elif self.modality == 'reaction':
            self.reaction_embed = ReactionRepresentation(embed_dim=self.embed_dim)

    def forward(self, data):
        """
        Run embedding based on modality type.
        """
        if self.modality == 'protein-sequence':
            embeddings, attention_mask = self.protein_embed(data)
            return embeddings, attention_mask
        elif self.modality == 'structure':
            embeddings = self.structure_embed(data)
            return embeddings
        elif self.modality == 'reaction':
            embeddings = self.reaction_embed(data)
            return embeddings
