import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from peft import LoraConfig, get_peft_model, TaskType
import re
import torch.nn.init as init

class ProteinSequenceEmbedding(nn.Module):
    def __init__(self, embed_dim=1024, stage="inference"):
        super(ProteinSequenceEmbedding, self).__init__()
        self.stage = stage
        model_path = "./model/ProtT5"
        # Load ProteinT5 tokenizer and encoder model
        print("Loading ProtT5")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        self.model = T5EncoderModel.from_pretrained(model_path)

        # Linear layer to project embedding dimension
        self.linear = nn.Linear(self.model.config.d_model, embed_dim)

        # Xavier initialization for linear layer
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)

        # Select target layers for LoRA finetuning (last 6 layers)
        target_modules = []
        for i in range(18, 24):  # Finetune only the last 6 layers
            target_modules.extend([
                f"encoder.block.{i}.layer.0.SelfAttention.q",
                f"encoder.block.{i}.layer.0.SelfAttention.k",
                f"encoder.block.{i}.layer.0.SelfAttention.v",
                f"encoder.block.{i}.layer.0.SelfAttention.o",  # Added: attention output projection
                f"encoder.block.{i}.layer.1.DenseReluDense.wi",  # Added: FFN first layer
                f"encoder.block.{i}.layer.1.DenseReluDense.wo",  # Added: FFN second layer
            ])


        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Feature extraction task
            r=8,  # LoRA low-rank parameter
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )

        # Freeze all original parameters first
        for name, param in self.model.named_parameters():
            param.requires_grad = False

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()  # Train mode, allowing LoRA gradient updates

        # Device setup (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.linear = self.linear.to(self.device)

    def forward(self, sequences):
        """
        Process protein sequences and return projected embeddings.
        sequences: list of amino-acid sequence strings.
        """
        # Preprocess: replace ambiguous amino acids and insert spaces
        sequences = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in sequences]

        # Convert sequences into model input format
        inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False)
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Compute embeddings with ProteinT5
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # Use last hidden state

        # Project to target embedding dimension
        adjusted_embeddings = self.linear(embeddings)

        if self.stage == "training":
            return adjusted_embeddings, attention_mask

        return embeddings, attention_mask

