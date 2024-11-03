import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab : int, n_embed : int, n_tokens : int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.PArameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        x = self.token_embedding(tokens)

        x += self.position_embedding

        return x
    
class CLIPLayer(nn.Module):

    def __init__(self, n_head : int, n_embed : int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        

class CLIP(nn.Module):

    def __init__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.Module([
            CLIPLayer(12, 768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens : torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers :
            state = layer(state)
        output = self.layernorm(state)

        return output
