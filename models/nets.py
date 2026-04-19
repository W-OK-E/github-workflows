"""
models/nets.py — LocoTransformer and LocoAttnResTransformer architectures.

Re-exports the key classes from torchrl.networks so that this package can
serve as the authoritative entry point for model architecture.  Class
definitions live in torchrl/networks/ (the copied RL library); this module
exposes the Loco-specific subset under a clean namespace.
"""

from torchrl.networks.base import LocoTransformerEncoder      # CNN encoder + state projector
from torchrl.networks.nets import (                            # transformer bodies
    LocoTransformer,
    SelfAttentionSubLayer,
    FFNSubLayer,
    AttnResLayer,
    LocoAttnResTransformer,
)

__all__ = [
    "LocoTransformerEncoder",
    "LocoTransformer",
    "SelfAttentionSubLayer",
    "FFNSubLayer",
    "AttnResLayer",
    "LocoAttnResTransformer",
]


if __name__ == "__main__":
    import torch

    # Quick smoke-test: build encoder + AttnRes model, run a forward pass.
    STATE_DIM = 84
    IMG_CHANNELS = 4
    BATCH = 2
    OBS_DIM = STATE_DIM + IMG_CHANNELS * 64 * 64

    encoder = LocoTransformerEncoder(
        in_channels=IMG_CHANNELS,
        state_input_dim=STATE_DIM,
        hidden_shapes=[256, 256],
        token_dim=64,
        two_by_two=False,
    )
    model = LocoAttnResTransformer(
        encoder=encoder,
        state_input_shape=STATE_DIM,
        visual_input_shape=(IMG_CHANNELS, 64, 64),
        output_shape=12,
        transformer_params=[[4, 128], [4, 128]],
        append_hidden_shapes=[256, 256],
        attn_res_heads=4,
        token_norm=True,
    )

    x = torch.randn(BATCH, OBS_DIM)
    out = model(x)
    print(f"LocoAttnResTransformer output shape: {out.shape}")
    assert out.shape == (BATCH, 12), f"Unexpected shape: {out.shape}"
    print("Smoke-test passed.")
