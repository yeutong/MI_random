# %%
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name, tokenize_and_concatenate
import torch.nn.functional as F

from functools import partial
import plotly.express as px
import numpy as np
from datasets import load_dataset

torch.set_grad_enabled(False)

from sae_lens import SparseAutoencoder
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
model = HookedTransformer.from_pretrained("gpt2-xl")
# %%
max_length = 256
openwebtext = load_dataset("stas/openwebtext-10k", split="train")
dataset = tokenize_and_concatenate(openwebtext, model.tokenizer, max_length=max_length)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
tokens = next(iter(data_loader))["tokens"]

# %%
# nev/gpt2_xl_saes-saex-test