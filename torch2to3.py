import io

import torch
from torch import nn

from models.max_sentence_embedding import Model, SentenceEncodingRNN

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def byte_convert(model: dict) -> dict:
    new_model = dict()
    for key in model:
        if type(key) == bytes:
            new_key = key.decode("utf-8")
        else:
            new_key = key
        new_model[new_key] = dict()
        if isinstance(model[key], dict):
            new_model[new_key] = byte_convert(model[key])
        else:
            new_model[new_key] = model[key]
    return new_model


def load_model(model_path: str) -> dict:
    with open(model_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
        return torch.load(buffer, map_location=device)


def save_model(model: nn.Module, state_dict: dict) -> None:
    model.load_state_dict(state_dict)
    model.eval()
    torch.save(model, type(model).__name__ + '.pth')


model = load_model('model_cpu.t7')

rnn = SentenceEncodingRNN(300, 256, 2)

save_model(Model(rnn, hidden=256), model.state_dict())
