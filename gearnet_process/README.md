# GearNet embedding pipeline (reproduction)

This folder builds the structure embeddings used by training: PDB files are turned into serialized graph objects in one HDF5, then GearNet runs on that HDF5 and writes a second HDF5 with datasets `ids` and `embeddings` (3072-dimensional vectors). That output matches what `train.py` expects via `--gearnet_h5_path`.

## Prerequisites

- Project layout: `./data` for PDBs and HDF5 outputs, `./model` for weights.
- GearNet weights for step 2: `./model/angle_gearnet_edge.pth` (same layout as the main RAMER release).

## 1. Environment

GearNet is provided through TorchDrug; install PyTorch, PyG wheels as needed, then TorchDrug.

**Recommended (manual)** — CUDA 12.1 + PyTorch 2.3 example; adjust for your stack:

```bash
pip install torch==2.3.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter torch-cluster -f https://pytorch-geometric.com/whl/torch-2.3.0+cu121.html
pip install torchdrug
conda install conda-forge::libstdcxx-ng
```

If importing TorchDrug fails with:

```text
ImportError: libXext.so.6: cannot open shared object file: No such file or directory
```

install the missing system libraries (Debian/Ubuntu example):

```bash
sudo apt-get update
sudo apt-get install libxrender1
```

**Alternative:** dedicated conda environment from this folder:

```bash
conda env create -f ./gearnet_process/gearnet.yml
conda activate <env_name_from_yml>
```

## 2. Prepare PDB files

Extract the PDB archive under the project (paths may differ; `set1_graph.py` only needs a directory of `.pdb` files):

```bash
tar -xzf ./data/pdb.tar.gz -C ./data
```

Point `--input_dir` at the folder that contains the `.pdb` files.

## 3. Step 1 — build graph HDF5 (`set1_graph.py`)

Reads every `.pdb` in `--input_dir`, builds TorchDrug graphs, and appends to `--output_h5` with datasets: `ids`, `graphs` (serialized), `graph_sizes`.

Example:

```bash
python ./gearnet_process/set1_graph.py \
  --input_dir ./data/pdb/pdb_data \
  --output_h5 ./data/graph.h5
```

Files that fail validation are skipped; their paths are printed at the end.

## 4. Step 2 — GearNet embeddings (`set2_gearnet_embedding.py`)

Loads graphs from the HDF5 produced in step 1, runs `./model/angle_gearnet_edge.pth`, and writes `--output_h5` with `ids` and `embeddings` (float32, shape N × 3072).

Example:

```bash
python ./gearnet_process/set2_gearnet_embedding.py \
  --input_h5 ./data/graph.h5 \
  --output_h5 ./data/gernet_embedding.h5 \
  --batch_size 8
```

Use the resulting file as `--gearnet_h5_path` when running `train.py`.
