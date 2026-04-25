import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import json
import torch.distributed as dist
import argparse
import os
import time
from torch.amp import GradScaler
import numpy as np
import random
import warnings
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForMaskedLM
sys.path.append(".Data2seq")
from Data2seq import BioData2Seq
import logging

####################### Logging, argument parsing, and DDP setup #######################################

def setup_logging(log_file='./training_t_position_loss.log'):
    # Create parent directory only when it is explicitly provided.
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def prepare():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        default=None,
        type=str,
        help="CUDA_VISIBLE_DEVICES override (default: keep existing environment setting)",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=24,
        type=int,
        metavar="N",
        help="number of batchsize",
    )
    parser.add_argument(
        "--seq_data",
        default="./data/uniprot_20W_struct_seq_reaction_121_without_new392.json",
        type=str,
        help="path to sequence data JSON",
    )
    parser.add_argument(
        "--reaction_data",
        default="./data/updated_rhea-reaction-smiles.json",
        type=str,
        help="path to reaction mapping JSON",
    )
    parser.add_argument(
        "--gearnet_embedding_path",
        default="./data/gernet_embedding",
        type=str,
        help="path to gearnet embedding directory",
    )
    parser.add_argument(
        "--log_file",
        default="./training_t_position_loss.log",
        type=str,
        help="path to training log file",
    )
    parser.add_argument(
        "--model_save_dir",
        default="./train_model",
        type=str,
        help="directory to save modality model checkpoints",
    )
    args = parser.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args

def set_seed(seed=3407):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable dynamic benchmark selection

# DDP initialization
def init_ddp(local_rank):
    # after this setup, tensors can be moved to GPU via `a = a.cuda()` rather than `a = a.to(local_rank)`
    torch.cuda.set_device(local_rank)
    os.environ["RANK"] = str(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

# Random generator for DDP workers
def get_ddp_generator(seed=3407):
    local_rank = int(os.environ["LOCAL_RANK"])
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g
########################################################################################

############################################## Dataset ####################################################
class BioDataset(Dataset):
    def __init__(self, json_file, rhea_smiles_mapping, graph_embedding_path):
        """
        Initialize dataset class.
        
        json_file: JSON file path for reaction data.
        rhea_smiles_mapping: Mapping dictionary from reaction IDs to SMILES.
        graph_embedding_path: Path to load gearNet embeddings.
        """
        # Load reaction data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Load SMILES mapped by RHEA ID
        with open(rhea_smiles_mapping, 'r') as f:
            self.rhea_smiles_mapping = json.load(f)
        
        # gearNet graph embedding path
        self.pdb_path = graph_embedding_path
        self.pdb_files = [] 

        # Build gearnet embedding paths from primaryAccession in JSON
        for item in self.data:
            primary_accession = item.get('primaryAccession')
            if primary_accession:
                pdb_file = os.path.join(self.pdb_path, f"{primary_accession}_graph_embedding.pt")
                self.pdb_files.append(pdb_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get protein sequence
        protein_sequence = item['sequence']  # Protein sequence
        language_text = item["language_text"]
        
        # Get reaction RHEA IDs
        rhea_ids = item.get('RHEA')
        
        # Get SMILES representation by RHEA ID
        if rhea_ids in self.rhea_smiles_mapping:
            smiles = self.rhea_smiles_mapping[rhea_ids]
                # Split reactants and products
            reactants, products = smiles.split('>>')
            reactant_smiles = reactants.split('.')  # Multiple reactants are separated by "."
            product_smiles = products.split('.')  # Multiple products are separated by "."
            reaction_smiles = (reactant_smiles, product_smiles)

        # Get corresponding gearnet embedding path
        gearnet_embedding_path = self.pdb_files[idx] if idx < len(self.pdb_files) else None
        
        # Load gearnet embedding tensor (shape: [3072])
        graph_embedding = torch.load(gearnet_embedding_path, weights_only=True)  # Read .pt file
        graph_embedding = graph_embedding.unsqueeze(0)  # Reshape to [1, 1, 3072]

        # Return batched fields
        return {
            'protein_sequence': protein_sequence,
            'language_text': language_text,
            'reaction_smiles': reaction_smiles,  # Reactant/product SMILES pair
            'gearnet_embedding': graph_embedding,  # Graph embedding tensor
        }
    
def custom_collate_fn(batch):
    protein_sequences = [item['protein_sequence'] for item in batch]
    texts = [item['language_text'] for item in batch]
    
    # Process reaction SMILES
    reaction_smiles = [item['reaction_smiles'] for item in batch]
    
    # Process gearnet embeddings
    gearnet_embeddings = [item['gearnet_embedding'] for item in batch]
    gearnet_embeddings = torch.cat(gearnet_embeddings, dim=0)  # Merge into [batch_size, 1, 3072]

    # No additional count processing needed here

    return {
        'protein_sequence': protein_sequences,
        'language_text': texts,
            'reaction_smiles': reaction_smiles,  # Reactant/product SMILES pair
            'gearnet_embedding': gearnet_embeddings,  # Processed embedding tensor
    }
###################################################################################################


########################################### Model saving ################################################
def save_model_and_tokenizer(protein_seq_tokenizer, structure_tokenizer, reaction_tokenizer, epoch, save_dir="/train_model"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(protein_seq_tokenizer.state_dict(), os.path.join(save_dir, f"seq_tokenizer_esm_{epoch}.pth"))
    torch.save(structure_tokenizer.state_dict(), os.path.join(save_dir, f"structure_tokenizer_{epoch}.pth"))
    torch.save(reaction_tokenizer.state_dict(), os.path.join(save_dir, f"reaction_tokenizer_{epoch}.pth"))
##################################################################################################


################################################################################################
class InfoNCELoss(nn.Module):
    def __init__(self, init_temp=1.0):
        super(InfoNCELoss, self).__init__()
        # Temperature factor is learnable and initialized by init_temp.
        self.temperature = nn.Parameter(torch.tensor(init_temp, dtype=torch.float32))  # Learnable parameter

    def forward(self, similarity_matrix):
        """
        Compute InfoNCE loss.
        
        similarity_matrix: (batch_size, batch_size), pairwise similarity matrix.
        labels: (batch_size), ground truth labels, usually 0 to batch_size-1.
        """

        self.temperature.data = torch.clamp(self.temperature.data, min=0.1)
        # Compute log-softmax over similarity matrix
        batch_size = similarity_matrix.size(0)

        # Scale similarity by temperature
        sim_matrix_scaled = similarity_matrix / self.temperature

        # Compute log probabilities
        log_prob = F.log_softmax(sim_matrix_scaled, dim=-1)

        # Build target index (diagonal alignment)
        target = torch.arange(batch_size).to(similarity_matrix.device)

        # InfoNCE loss
        loss = -log_prob[torch.arange(batch_size), target].mean()

        return loss
################################################################################################

########################################## Training helpers #############################################
def create_labels(protein_sequences):
  
    label_protein_to_others = torch.arange(len(protein_sequences)).cuda()
    labels_protein_to_reaction = torch.arange(len(protein_sequences)).cuda()

    # Return labels and ensure they are CUDA tensors
    return label_protein_to_others.clone().detach().cuda(), labels_protein_to_reaction.clone().detach().cuda()


def calculate_contrastive_loss(similarity_matrix, labels, modality = "",):

    # Cross entropy for sequence/structure/text alignment
    criterion_seq_struct_text = nn.CrossEntropyLoss()

    # Cross entropy for sequence/molecule alignment
    criterion_molecule = nn.CrossEntropyLoss()

    # Keep label handling consistent with current implementation
    if modality == "seq_to_molecule":
        protein_to_reaction_loss = criterion_molecule(similarity_matrix, labels)
        reaction_to_protein_loss = criterion_molecule(similarity_matrix.T, labels)
        loss = (protein_to_reaction_loss+reaction_to_protein_loss) / 2

    else:
        protein_to_another_loss = criterion_seq_struct_text(similarity_matrix, labels)
        # Structure-to-protein similarity loss (no label transpose needed)
        another_to_protein_loss = criterion_seq_struct_text(similarity_matrix.T, labels) 

        loss = (protein_to_another_loss + another_to_protein_loss) / 2

    return loss

##############################################################################################

#################################### Training function ###################################################
def train(protein_seq_tokenizer, structure_tokenizer, reaction_tokenizer, train_loader, optimizer, train_loss, scaler, log_file='./training_t_position_loss.log'):
    setup_logging(log_file)
    
    # Set tokenizer and encoder modules to train mode
    protein_seq_tokenizer.train()
    structure_tokenizer.train()
    reaction_tokenizer.train()

    # Track losses during training
    total_loss = torch.tensor(0.00).cuda()  # Total loss for current epoch
    struct_loss = torch.tensor(0.00).cuda()
    reaction_loss = torch.tensor(0.00).cuda()

    for batch in tqdm(train_loader):
        protein_sequences = batch['protein_sequence']  # Protein sequence modality
        gearnet_embedding_batch = batch['gearnet_embedding']  # Structure modality (frozen gearnet features)
        reaction_smiles = batch['reaction_smiles']  # Reaction modality

        # Forward pass
        protein_embeddings, protein_mask = protein_seq_tokenizer(protein_sequences)
        structure_embeddings = structure_tokenizer(gearnet_embedding_batch)
        reaction_embeddings = reaction_tokenizer(reaction_smiles)
        
        protein_avg_embedding = torch.sum(protein_embeddings * protein_mask.unsqueeze(-1), dim=1) / torch.sum(protein_mask, dim=1, keepdim=True)
        structure_avg_embedding = structure_embeddings
        reaction_avg_embeddings = reaction_embeddings

        # Sequence-to-structure and sequence-to-reaction similarity
        protein_to_structure_similarity_matrix = F.cosine_similarity(protein_avg_embedding.unsqueeze(1), structure_avg_embedding.unsqueeze(0), dim=-1)
        protein_to_reaction_similarity_matrix = F.cosine_similarity(protein_avg_embedding.unsqueeze(1), reaction_avg_embeddings.unsqueeze(0), dim=-1)

        # Create labels for similarity matrices
        label_protein_to_others, labels_protein_to_reaction = create_labels(protein_sequences)
        
        # Compute losses
        protein_to_reaction_loss = train_loss(protein_to_reaction_similarity_matrix)
        reaction_to_protein_loss = train_loss(protein_to_reaction_similarity_matrix.T)
        loss_seq_and_reaction = (protein_to_reaction_loss + reaction_to_protein_loss) / 2

        protein_to_another_loss = train_loss(protein_to_structure_similarity_matrix)
        another_to_protein_loss = train_loss(protein_to_structure_similarity_matrix.T)
        loss_seq_and_structure = (protein_to_another_loss + another_to_protein_loss) / 2

        loss = 0.4 * loss_seq_and_structure + 0.6 * loss_seq_and_reaction

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss
        struct_loss += loss_seq_and_structure
        reaction_loss += loss_seq_and_reaction

    avg_loss = total_loss / len(train_loader)
    avg_struct = struct_loss / len(train_loader)
    avg_reaction = reaction_loss / len(train_loader)

    dist.reduce(avg_loss, 0, op=dist.ReduceOp.SUM)  # Reduce mean loss from all processes to rank 0
    dist.reduce(avg_struct, 0, op=dist.ReduceOp.SUM) 
    dist.reduce(avg_reaction, 0, op=dist.ReduceOp.SUM) 

    local_rank = dist.get_rank()  # Get current process rank
    if local_rank == 0:  # Print only on rank 0
        avg_loss = avg_loss / dist.get_world_size()
        struct_avg_loss = avg_struct / dist.get_world_size()
        reaction_avg_loss = avg_reaction / dist.get_world_size()

        # Get and record temperature value
        temperature_value = train_loss.temperature.item()

        # Print and log metrics
        print(f"Training set: Loss: {avg_loss:.4f}, struct Loss:{struct_avg_loss:.4f}, reaction Loss:{reaction_avg_loss:.4f}, Temperature: {temperature_value:.4f}")
        logging.info(f"Training set: Loss: {avg_loss:.4f}, struct Loss:{struct_avg_loss:.4f}, reaction Loss:{reaction_avg_loss:.4f}, Temperature: {temperature_value:.4f}")



def print_model_params(model, model_name):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters in {model_name}: {total_params}")
    print(f"Trainable parameters in {model_name}: {trainable_params}")



def load_model_weights(model, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        # Remove "module." prefix when loading from DDP checkpoints
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        
        model.load_state_dict(checkpoint)
        print(f"Successfully loaded model weights: {checkpoint_path}")
    else:
        print(f"Model file not found: {checkpoint_path}")
    return model


def main(args):
    # Initialize DDP first
    local_rank = int(os.environ["LOCAL_RANK"])
    init_ddp(local_rank)
    protein_seq_tokenizer = BioData2Seq(
        modality='protein-sequence',
        embed_dim=1024,
        protein_stage="training"
    ).cuda()
    structure_tokenizer = BioData2Seq(modality= 'structure', embed_dim=1024).cuda()
    reaction_tokenizer = BioData2Seq(modality='reaction', embed_dim=1024).cuda()


    # Loss function
    info_nce_loss = InfoNCELoss(init_temp=1.0)

    print_model_params(protein_seq_tokenizer, "protein_seq_tokenizer")
    print_model_params(structure_tokenizer, "structure_tokenizer")
    print_model_params(reaction_tokenizer, "reaction_tokenizer")

    print("Tokenizer and encoder models loaded")

    # Wrap with DDP, each process uses its own GPU
    protein_seq_tokenizer = nn.parallel.DistributedDataParallel(protein_seq_tokenizer, device_ids=[local_rank])
    # find_unused_parameters=True)
    structure_tokenizer = nn.parallel.DistributedDataParallel(structure_tokenizer, device_ids=[local_rank])
    reaction_tokenizer = nn.parallel.DistributedDataParallel(reaction_tokenizer, device_ids=[local_rank],find_unused_parameters=True)

    optimizer = optim.AdamW(
        [
            {'params': filter(lambda p: p.requires_grad, protein_seq_tokenizer.parameters())},
            {'params': filter(lambda p: p.requires_grad, structure_tokenizer.parameters())},
            {'params': filter(lambda p: p.requires_grad, reaction_tokenizer.parameters())},
            {'params': info_nce_loss.temperature, 'lr': 2e-5}  # Separate LR for temperature parameter
        ],
        lr=0.0001  # Default learning rate
    )

    # Build dataset from CLI-provided paths.
    json_file = args.seq_data
    rhea_smiles_file = args.reaction_data
    graph_embedding_path = args.gearnet_embedding_path

    train_dataset = BioDataset(json_file=json_file, rhea_smiles_mapping=rhea_smiles_file, graph_embedding_path=graph_embedding_path)
    # Create one DataLoader per GPU and keep DDP data partition aligned
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    g = get_ddp_generator() # Random generator for DDP workers
    # Build dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        shuffle=False,
        pin_memory=True,
        generator=g,
        collate_fn=custom_collate_fn
        )
    scaler = GradScaler(enabled=False)
    # Start training
    for epoch in range(args.epochs):
        if local_rank == 0:  ### avoid redundant printing for each process
            print(f"begin training of epoch {epoch + 1}/{args.epochs}")
        train_loader.sampler.set_epoch(epoch) # Ensure per-epoch shard reshuffle under DDP
        # Run one training epoch
        train(
           protein_seq_tokenizer = protein_seq_tokenizer,
           structure_tokenizer = structure_tokenizer,
           reaction_tokenizer = reaction_tokenizer,
           train_loader = train_loader,
           optimizer = optimizer,
           scaler = scaler,
           train_loss = info_nce_loss, 
           log_file = args.log_file
           )
        # print(info_nce_loss.temperature)
        # Save once per epoch on rank 0
        if local_rank == 0:
             save_model_and_tokenizer(protein_seq_tokenizer, structure_tokenizer, reaction_tokenizer, epoch, args.model_save_dir)
    dist.destroy_process_group() # Clean up process group

if __name__ == "__main__":
    args = prepare()
    set_seed()
    time_start = time.time()
    main(args)
    time_elapsed = time.time() - time_start
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print(f"\ntime elapsed: {time_elapsed:.2f} seconds")