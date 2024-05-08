# %%
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
import torch.nn.functional as F

from functools import partial
import plotly.express as px
import numpy as np
from datasets import load_dataset

torch.set_grad_enabled(False)

from sae_lens import SparseAutoencoder
from sae_lens.toolkit.pretrained_saes import get_gpt2_res_jb_saes


# %%
model = HookedTransformer.from_pretrained("gpt2-small")
# %%
# model
# %%
prompt = "Access and plot the attention pattern of head L2H4 on the prompt"
logit, cache = model.run_with_cache(prompt)
# %%
layer = 2
head = 4
L2_atten_pattern = get_act_name("pattern", layer)
L2H4_atten_pattern = cache[L2_atten_pattern][0, head]  # shape: 16, 16

# %%
fig = px.imshow(
    L2H4_atten_pattern.cpu(),
    title=f"L{layer}H{head} Attention Pattern",
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0,
)

# fig.show()


# %%
def zero_ablate_mlp(activation, hook):
    # activation: [batch, pos, d_model]
    activation[:, :, :] = torch.zeros_like(activation)
    return activation


hook_name = get_act_name("mlp_out", 0)
logits = model.run_with_hooks(
    prompt, fwd_hooks=[(hook_name, zero_ablate_mlp)], return_type="logits"
)
# %%
rand_logits = torch.randn_like(logits)
# %%
rand_2d = np.random.rand(2, 16)
# %%
# px.scatter(rand_2d, x=rand_2d[0], y=rand_2d[1])


# %%
hook_name = get_act_name("resid_pre", 8)
saes, sparsities = get_gpt2_res_jb_saes(hook_name)
# %%
print(saes.keys())
# %%
sae = saes[hook_name]
# %%
# %%
sae.W_dec.shape
# %%
# reconstruct activation, and compared with the real one
real_act = cache[hook_name].cpu()  # shape d_model
# %%
sae_in = real_act - (sae.b_dec * sae.cfg.apply_b_dec_to_input)
hidden_pre = (sae_in @ sae.W_enc) + sae.b_enc
feature_act = F.relu(hidden_pre)
sae_out = (feature_act @ sae.W_dec) + sae.b_dec
sae_out

# %%
for i, _ in sae.named_parameters():
    print(i)
# %%
feature_act_real, hidden_pre_real = sae._encode_with_hidden_pre(real_act)
# %%
sae_out_real = sae.decode(feature_act_real)
# %%
# calc reconstruction error
recon_error = F.mse_loss(sae_out_real, real_act)
print(recon_error)

recon_error = F.mse_loss(sae_out, real_act)
print(recon_error)


# %%

sae_out, feature_acts, loss, mse_loss, l1_loss, ghost_grad_loss = sae(real_act)
# %%
sae_out.shape
# %%
feature_acts.shape
# %%
loss.shape
# %%
mse_loss.shape
# %%
l1_loss.shape
# %%
# load dataset from huggingface
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-small")
dataset = load_dataset("stas/openwebtext-10k")
# %%
model.generate("你好")
# %%
