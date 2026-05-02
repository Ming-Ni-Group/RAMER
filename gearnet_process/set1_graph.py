import argparse
import io
import os

import h5py
import numpy as np
import torch
from torchdrug import data, layers, transforms
from torchdrug.layers import geometry
from tqdm import tqdm


def initialize_hdf5(hdf5_path):
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("ids", (0,), maxshape=(None,), dtype="S100")
        vlen_uint8 = h5py.vlen_dtype(np.dtype("uint8"))
        f.create_dataset("graphs", (0,), maxshape=(None,), dtype=vlen_uint8)
        f.create_dataset("graph_sizes", (0,), maxshape=(None,), dtype=np.int64)


def append_to_hdf5(hdf5_path, new_ids, new_graph_bytes):
    with h5py.File(hdf5_path, "a") as f:
        old_size = f["ids"].shape[0]
        new_size = old_size + len(new_ids)
        f["ids"].resize((new_size,))
        f["graphs"].resize((new_size,))
        f["graph_sizes"].resize((new_size,))
        f["ids"][old_size:new_size] = np.array(new_ids, dtype="S")
        f["graphs"][old_size:new_size] = new_graph_bytes
        f["graph_sizes"][old_size:new_size] = np.array(
            [len(x) for x in new_graph_bytes], dtype=np.int64
        )


def serialize_graph(graph_obj):
    buffer = io.BytesIO()
    torch.save(graph_obj, buffer)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


class ProteinProcessor:
    def __init__(self, pdb_directory, output_h5, graph_construction_model, transform):
        self.pdb_directory = pdb_directory
        self.output_h5 = output_h5
        self.graph_construction_model = graph_construction_model
        self.transform = transform
        self.unprocessed_files = []
        output_parent = os.path.dirname(output_h5) or "."
        os.makedirs(output_parent, exist_ok=True)

    def process_pdb_files(self):
        pdb_files = sorted([f for f in os.listdir(self.pdb_directory) if f.endswith(".pdb")])
        initialize_hdf5(self.output_h5)

        for pdb_file in tqdm(pdb_files, desc="Build graph h5"):
            pdb_file_path = os.path.join(self.pdb_directory, pdb_file)
            primary_accession = os.path.splitext(pdb_file)[0]
            try:
                protein = data.Protein.from_pdb(
                    pdb_file_path,
                    atom_feature="position",
                    bond_feature="length",
                    residue_feature="symbol",
                )
                packed = data.Protein.pack([protein])
                graph = self.graph_construction_model(packed)
                graph = data.Protein.pack([graph]).cpu()
                graph = self.transform({"graph": graph})["graph"]
                graph_bytes = serialize_graph(graph)
                append_to_hdf5(self.output_h5, [primary_accession], [graph_bytes])
            except ValueError as e:
                print(f"Error processing {pdb_file_path}: {e}. Skipping this file.")
                self.unprocessed_files.append(pdb_file_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PDB files into graph objects and save into one HDF5."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/fsas/A_RAMMER/data/pdb/pdb_data",
        help="Input directory containing .pdb files",
    )
    parser.add_argument(
        "--output_h5",
        type=str,
        default="/root/fsas/A_RAMMER/data/graph.h5",
        help="Output HDF5 path for processed graph objects",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    graph_construction_model = layers.GraphConstruction(
        node_layers=[geometry.AlphaCarbonNode()],
        edge_layers=[
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
            geometry.SequentialEdge(max_distance=2),
        ],
        edge_feature="gearnet",
    )
    transform = transforms.ProteinView(view="residue")
    processor = ProteinProcessor(
        args.input_dir, args.output_h5, graph_construction_model, transform
    )
    processor.process_pdb_files()
    print(f"Done. Graph HDF5 saved to: {args.output_h5}")
    print("Unprocessed files:", processor.unprocessed_files)


if __name__ == "__main__":
    main()