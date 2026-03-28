import tiktoken
import torch
import torch.nn as nn

enc = tiktoken.get_encoding("cl100k_base")
d_model = 16 #d model is the size of an embedding vector for each token
embedding = nn.Embedding(enc.n_vocab, d_model)

def get_token_ids(text: str) -> torch.Tensor:
    token_ids = enc.encode(text)
    return torch.tensor(token_ids, dtype=torch.long)

def generate_embeddings(token_ids: torch.Tensor) -> torch.Tensor:
    return embedding(token_ids)

# test
if __name__ == "__main__":
    text = "Hello, how are you?"
    token_ids = get_token_ids(text)
    print("Token IDs:", token_ids)

    embeddings = generate_embeddings(token_ids)
    print("Embeddings shape:", embeddings.shape)