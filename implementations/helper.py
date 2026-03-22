import tiktoken
import torch
import torch.nn as nn

#1. Text -> Tokens
#split on whitespace, punctuation, and special characters
enc = tiktoken.get_encoding("cl100k_base")
d_model = 16
embedding = nn.Embedding(enc.n_vocab, d_model)

def get_token_ids(text: str) -> torch.Tensor:
    token_ids = enc.encode(text)
    return torch.tensor(token_ids, dtype=torch.long)

def generate_embeddings(token_ids: torch.Tensor) -> torch.Tensor:
    return embedding(token_ids)