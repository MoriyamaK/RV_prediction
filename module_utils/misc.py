import random
import numpy as np
import torch
from io import BytesIO

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

def copy_model(model):

    with BytesIO() as f:
        torch.save(model, f)
        f.seek(0)
        model_copy = torch.load(f)

    return model_copy
