# Training Configs

This directory is the home for YAML-based training configs.

Intended convention:

- one YAML file per training config
- file names are stable and readable
- `VERSION` inside the YAML is required
- the training notebook should load one config from here and not own its own hyperparameters

Current status:

- `configs/coord_to_image_unet_all_current_data_v1.yaml` is the explicit all-current-data baseline config
- `configs/coord_to_image_unet_v1.yaml` is the single-dataset baseline config
- `configs/coord_to_image_unet_smoke.yaml` is the small verification config
- `configs/pointing_cvae_v1.yaml` is the latent posterior/prior training config
- `configs/pointing_cvae_smoke.yaml` is the small latent-path verification config
- `configs/pointing_cvae_march_small_v1.yaml` is the March-only `SMALL` latent config for faster iteration
- `configs/pointing_cvae_march_big_v1.yaml` is the March-only `BIG` latent config for longer `128x128` training
- `scripts/train_from_config.py` is the primary command-line entrypoint
- `model/workspace.ipynb` loads one config and uses the same Python training path
- committed configs include a `wandb:` section so training logs to Weights & Biases by default

See:

- `docs/training_configs.md`
