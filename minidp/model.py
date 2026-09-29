"""Tiny double-precision token model used for the verifiable prototype.

Embedding(vocab=4, dim=5) -> Tanh-MLP -> 4 logits per sequence position.
float64 is used on purpose: the distributed SUM-all-reduce path must reproduce
the single-process full-batch oracle to very tight tolerance on CPU/Gloo.
"""
import torch

torch.set_num_threads(1)

VOCAB = 4


class TokenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB, 5)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 7),
            torch.nn.Tanh(),
            torch.nn.Linear(7, 6),
            torch.nn.Tanh(),
            torch.nn.Linear(6, VOCAB),
        )

    def forward(self, tokens):  # tokens: [B, S] long or [S] long
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        h = self.embed(tokens)
        out = self.net(h)
        return out.squeeze(0) if squeeze else out


def build_model():
    model = TokenModel().to(torch.float64)
    model.eval()
    return model


def init_model(seed):
    torch.manual_seed(seed)
    return build_model()
