# %%

# %%
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify.sparsify import SaeConfig, SaeTrainer, TrainConfig
from sparsify.sparsify.data import chunk_and_tokenize

MODEL = "EleutherAI/pythia-70m"
dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=512)
tokenized = tokenized.shuffle(seed=42)

gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

cfg = TrainConfig(
    SaeConfig(expansion_factor=8, k=4), 
    batch_size=16,
    layers=[4]
)
trainer = SaeTrainer(cfg, tokenized, gpt)

trainer.fit()
# %%