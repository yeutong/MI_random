# %%
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name


from functools import partial
import plotly.express as px
import numpy as np

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
