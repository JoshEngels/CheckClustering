# %%

# %%
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify.sparsify import SaeConfig, SaeTrainer, TrainConfig
from sparsify.sparsify.data import chunk_and_tokenize

MODEL = "openai-community/gpt2"
dataset = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample",
    split="train",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=512)


gpt = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map={"": "cuda"},
    torch_dtype=torch.bfloat16,
)

cfg = TrainConfig(
    SaeConfig(expansion_factor=4, k=4), 
    batch_size=16,
    layers=[8]
)
trainer = SaeTrainer(cfg, tokenized, gpt)

trainer.fit()
# %%