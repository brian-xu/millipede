import io

import torch
from torch import nn

from models.max_sentence_embedding import create

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(model_path: str) -> dict:
    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        return torch.load(buffer, map_location=device)


def save_model(model: nn.Module, state_dict: dict) -> None:
    model.load_state_dict(state_dict)
    model.eval()
    torch.save(model, type(model).__name__ + '.pth')


model = load_model('model_cpu.t7')

save_model(create(), model.state_dict())
