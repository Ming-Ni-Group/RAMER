import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm

sys.path.append("./Data2seq")
from Data2seq import BioData2Seq


EMBED_DIM = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAMMER embeddings from FASTA input.")
    parser.add_argument(
        "--input_name",
        required=True,
        type=str,
        help="Input file base name under ./input (for example: NEW-392).",
    )
    parser.add_argument(
        "--model_path",
        default="./model/RAMMER.pth",
        type=str,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--save_interval",
        default=1000,
        type=int,
        help="Save to HDF5 every N batches.",
    )
    parser.add_argument(
        "--input_dir",
        default="./input",
        type=str,
        help="Directory that stores FASTA/FAA files.",
    )
    parser.add_argument(
        "--output_dir",
        default="./RAMMER_embedding",
        type=str,
        help="Directory for generated HDF5 files.",
    )
    return parser.parse_args()


def resolve_input_fasta(input_name, input_dir):
    fasta_candidate = os.path.join(input_dir, f"{input_name}.fasta")
    faa_candidate = os.path.join(input_dir, f"{input_name}.faa")
    if os.path.exists(fasta_candidate):
        return fasta_candidate
    if os.path.exists(faa_candidate):
        return faa_candidate
    return None


def load_fasta_sequences(fasta_file):
    ids = []
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        ids.append(record.id)
        sequences.append(str(record.seq))
    return ids, sequences


def initialize_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("ids", (0,), maxshape=(None,), dtype="S100")
        f.create_dataset("embeddings", (0, EMBED_DIM), maxshape=(None, EMBED_DIM), dtype=np.float32)
    print(f"Initialized HDF5 file: {hdf5_path}")


def append_to_hdf5(hdf5_path, new_ids, new_embeddings):
    with h5py.File(hdf5_path, "a") as f:
        old_size = f["embeddings"].shape[0]
        new_size = old_size + len(new_embeddings)
        f["ids"].resize((new_size,))
        f["embeddings"].resize((new_size, EMBED_DIM))
        f["ids"][old_size:new_size] = np.array(new_ids, dtype="S")
        f["embeddings"][old_size:new_size] = np.array(new_embeddings, dtype=np.float32)
    print(f"Saved {len(new_embeddings)} records to HDF5.")


def main():
    args = parse_args()
    start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fasta_path = resolve_input_fasta(args.input_name, args.input_dir)
    if fasta_path is None:
        print(
            f"Input path is invalid. Neither {os.path.join(args.input_dir, args.input_name + '.fasta')} "
            f"nor {os.path.join(args.input_dir, args.input_name + '.faa')} exists."
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)
    hdf5_path = os.path.join(args.output_dir, f"{args.input_name}.h5")

    print(f"Using device: {device}")
    print(f"Input FASTA/FAA file: {fasta_path}")
    print(f"Output HDF5 file: {hdf5_path}")
    print(f"Model checkpoint: {args.model_path}")

    ids, sequences = load_fasta_sequences(fasta_path)
    print(f"Loaded {len(sequences)} sequences.")

    model = BioData2Seq(modality="protein-sequence", embed_dim=EMBED_DIM).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    print("Loading RAMMER checkpoint weights...")
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    initialize_hdf5(hdf5_path)

    all_embeddings = []
    batch_ids = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), args.batch_size)):
            batch_seqs = sequences[i : i + args.batch_size]
            batch_ids.extend(ids[i : i + args.batch_size])

            protein_embeddings, protein_mask = model(batch_seqs)
            protein_avg_embedding = torch.sum(
                protein_embeddings * protein_mask.unsqueeze(-1), dim=1
            ) / torch.sum(protein_mask, dim=1, keepdim=True)

            all_embeddings.extend(protein_avg_embedding.cpu())

            if (i // args.batch_size + 1) % args.save_interval == 0:
                append_to_hdf5(hdf5_path, batch_ids, all_embeddings)
                all_embeddings = []
                batch_ids = []

    if all_embeddings:
        append_to_hdf5(hdf5_path, batch_ids, all_embeddings)

    print("Embedding generation completed.")
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
