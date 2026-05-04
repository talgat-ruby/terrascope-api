"""Slimmed inference-only fork of htcr/sam_road (MIT, CVPRW 2024).

Vendored to avoid pulling Lightning/wandb/torchmetrics/addict/networkx/igraph
into core2 just to load a checkpoint. Only the pieces needed to run
`infer_masks_and_img_features` + `infer_toponet` and turn their masks into a
node/edge graph are kept; training, ablations, metrics, LoRA, SAM-decoder
variants, and visualization helpers were removed.

Upstream: https://github.com/htcr/sam_road
"""
