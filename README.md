# RAMER

RAMER is a pretrained reaction-aware multimodal AI model that integrates protein sequence, structure, and catalytic reaction information to improve enzyme function annotation, including strong recall at fine-grained EC levels.

At large scale, RAMER has been applied to billions of proteins across diverse environments and supports function-driven mining of valuable enzymes by linking sequence/structure signals with catalytic behavior. This repository provides practical scripts and workflows for:

- Zero-shot EC prediction (`top1` and `max-separation` strategies)
- Binary enzyme/non-enzyme classification on top of RAMER embeddings
- Distributed training reproduction (DDP with `torchrun`)

Project assets (`Background_library`, `data`, and `model`) are available on Hugging Face: [PengJiaMa123/RAMER](https://huggingface.co/PengJiaMa123/RAMER).

## Repository Structure

- `set1_get_RAMER_embedding.py`: generate RAMER embeddings from FASTA/FAA input
- `set2_top1_zero_shot.py`: zero-shot EC prediction with top-1 retrieval
- `set2_max_sep_zero_shot.py`: zero-shot EC prediction with dynamic max-separation selection
- `eval_ec_csv_with_background_dict.py`: EC evaluation script based on background dictionary labels
- `binary_enzyme_classifier.py`: enzyme/non-enzyme binary classifier from RAMER embeddings
- `train.py`: DDP training script
- `Data2seq/`: sequence/structure/reaction encoders and fusion components

## Environment Setup

### Option 1: Automatic install

```bash
git clone https://github.com/Ming-Ni-Group/RAMER.git
cd ./RAMER

# Create environment directly from ramer.yml
conda env create -f ramer.yml
conda activate ramer
```

### Option 2: Manual install (recommended)

```bash
git clone https://github.com/Ming-Ni-Group/RAMER.git
cd ./RAMER
conda create -n ramer python=3.10 -y
conda activate ramer

# Install PyTorch according to your CUDA/CPU environment
# Example (edit based on your machine): pip install torch torchvision torchaudio

pip install transformers tqdm sentencepiece protobuf scikit-learn h5py biopython xgboost peft
```

## Data and Model Preparation

Download project resources from [PengJiaMa123/RAMER](https://huggingface.co/PengJiaMa123/RAMER) and place them under the project root:

- `./Background_library`
- `./data`
- `./model`

### Training-only note

To reproduce training, first unzip:

- `./data/gernet_embedding.zip`

to get:

- `./data/gernet_embedding` (or pass a custom path via `--gearnet_embedding_path`)

For inference-only usage, this GearNet directory is not required.

## Inference Pipeline

### 1) Generate RAMER embeddings

Input only the file base name. The script resolves:

- `./input/<input_name>.fasta` first
- then `./input/<input_name>.faa`

```bash
python ./set1_get_RAMER_embedding.py --input_name NEW-392
```

Output:

- `./output/NEW-392.h5`

> If needed, you can override defaults such as `--model_path`, `--batch_size`, and `--save_interval`.

### 2) Zero-shot EC prediction (Top-1)

```bash
python ./set2_top1_zero_shot.py \
  --test_name NEW-392 \
  --batch_size 8000 \
  --background_library_h5 ./Background_library/clean100_set.h5 \
  --background_library_dict ./Background_library/clean100_set_dict.json
```

Output:

- `./output/NEW-392_top1.csv`

### 3) Zero-shot EC prediction (Max-separation)

```bash
python ./set2_max_sep_zero_shot.py \
  --test_name NEW-392 \
  --batch_size 8000 \
  --background_library_h5 ./Background_library/clean100_set.h5 \
  --background_library_dict ./Background_library/clean100_set_dict.json
```

Output:

- `./output/NEW-392_max-sep.csv`

### Background dictionary choice

- Use `clean100_set_dict.json` to reproduce comparison settings aligned with baseline training splits.
- Use `clean100_set_dict_updated_2025.json` when you want updated EC annotations.

## EC Evaluation

Use:

- `./eval_ec_csv_with_background_dict.py`

Prediction CSVs for RAMER and other methods on `new392` / `ram255` are located in:

- `./data/test_data`

## Enzyme / Non-enzyme Classification

This task requires RAMER embedding `.h5` files under `./RAMER_embedding`.

```bash
python ./binary_enzyme_classifier.py --input_name NEW-392
```

Output:

- `./output/NEW-392_enzyme_classifier.csv`

CSV columns:

- `test_id`
- `is_enzyme` (`0` or `1`)
- `enzyme_probability`

## Training Reproduction (DDP)

```bash
torchrun --nproc_per_node=8 train.py \
  --seq_data ./data/uniprot_20W_struct_seq_reaction_121_without_new392.json \
  --reaction_data ./data/updated_rhea-reaction-smiles.json \
  --gearnet_embedding_path ./data/gernet_embedding \
  --log_file ./training_t_position_loss.log \
  --model_save_dir ./train_model \
  --epochs 50 \
  --batch_size 24
```

## License

The source code in this repository is licensed under the MIT License.
See `LICENSE`.

## Third-Party Base Models

RAMER relies on third-party base models (ProtTrans/ProtT5, MolT5, and GearNet).
These models and their checkpoints remain under their original upstream licenses.
See `THIRD_PARTY_MODELS.md` for source links, license links, and usage scope
within this project.

