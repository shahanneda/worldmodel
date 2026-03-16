# Training Config System

## Status

The repo now has a working config-driven training path:

- training configs live in `configs/`
- the CLI entrypoint is `scripts/train_from_config.py`
- the notebook loads one YAML config and calls the same Python training path
- checkpoints now include `VERSION` in the filename and carry the resolved config metadata
- Weights & Biases logging is part of the shared training path

The refactor is not "done forever", but the YAML config system is now the intended source of truth for training runs.

## Why this change is needed

The current training path still spreads experiment state across:

- notebook cells in `model/workspace.ipynb`
- keyword arguments passed into `model/train.py`
- loosely structured `extra` checkpoint metadata
- checkpoint filename conventions

That is workable for a first pass, but it is not robust enough for:

- repeatable experiments
- multiple training setups living side by side
- safe checkpoint resume behavior
- future backward compatibility
- notebook-driven exploration without notebook-owned hyperparameters

The new source of truth should be one YAML file per training setup inside `configs/`.

## Goals

- Each training config is a YAML file in `configs/`.
- Each YAML file fully describes one training run family.
- The notebook becomes a thin runner and visualizer for a loaded config.
- The user-provided `VERSION` field is required and is prefixed into checkpoint names.
- The code validates configs strictly so typos or missing keys fail early.
- Checkpoints carry enough metadata to reproduce the run without trusting notebook state.
- Older config schemas can be upgraded forward later through explicit migration code.

## Working commands

Validate a config without training:

```bash
/venv/main/bin/python3 scripts/train_from_config.py configs/pointing_cvae_v1.yaml --dry-run
```

Run the committed smoke config:

```bash
/venv/main/bin/python3 scripts/train_from_config.py configs/pointing_cvae_smoke.yaml
```

Training auto-loads the repo-root `.env` before initializing W&B, so `WANDB_API_KEY` does not need to be manually sourced for normal training runs.

Run a config with a few overrides:

```bash
/venv/main/bin/python3 scripts/train_from_config.py configs/pointing_cvae_v1.yaml \
  --set VERSION=finger_xy_cvae_debug_v1 \
  --set model.image_size=32 \
  --set model.base_channels=8 \
  --set model.latent_dim=16 \
  --set training.epochs=1
```

## Non-goals

- Introducing a large config framework just for indirection.
- Hiding important defaults in many layers.
- Allowing silent fallback when a config key is wrong.
- Keeping notebook constants as a second source of truth.

## Core design

### 1. `configs/` is the experiment registry

The repo should have a committed `configs/` directory.

- One YAML file = one named training config.
- File names should be stable and readable, for example `coord_to_image_unet_v1.yaml`.
- Different model kinds should live side by side in the same registry, for example `coord_to_image_unet_v1.yaml` and `pointing_cvae_v1.yaml`.
- Once a config has been used for a real run, treat it as immutable.
- If behavior changes materially, create a new config file or bump `VERSION`.

### 2. Separate schema version from experiment version

There are two different version concepts and both are needed:

- `config_schema`
  - version of the YAML structure itself
  - used for backward compatibility and migration code
- `VERSION`
  - user-chosen experiment/version label
  - used in checkpoint naming and artifact grouping

These are not interchangeable.

Example:

```yaml
config_schema: 1
VERSION: finger_xy_baseline_v1
```

### 3. Typed config loading, not ad hoc dict access

The training code should load YAML into a validated typed object, not pass raw nested dicts everywhere.

Target shape:

- `model/config.py`
  - load YAML
  - normalize paths
  - validate required fields
  - reject unknown keys by default
  - upgrade old `config_schema` payloads to the latest schema
  - expose a typed `TrainingConfig`

That loader becomes the only supported way to read a training config.

### 4. One source of truth for each parameter

Avoid duplicate parameters that can drift.

For example:

- `image_size` should not live independently in both loader settings and model settings
- `run_name` should not exist in one place in the notebook and another in checkpoint code
- resume behavior should not be hidden behind a notebook boolean plus filename matching

If a value is conceptually global to the run, define it once and derive the rest.

## Proposed directory layout

```text
configs/
  README.md
  coord_to_image_unet_all_current_data_v1.yaml
  coord_to_image_unet_v1.yaml
  pointing_cvae_v1.yaml

model/
  config.py
  train.py
  checkpoints.py
  workspace.ipynb
  splits/
  artifacts/
```

## Proposed config structure

The canonical draft example lives at:

- `configs/pointing_cvae_v1.yaml`

The top-level sections should be explicit and stable:

- `config_schema`
- `VERSION`
- `experiment`
- `paths`
- `data`
- `splits`
- `loader`
- `model`
- `training`
- `checkpointing`
- `reproducibility`
- `notebook`
- `wandb`

### Data section

The config should describe **all** datasets used for training, not just a root directory.

Preferred shape:

- explicit `datasets` list
- each dataset entry names the processed directory
- path references are repo-relative in YAML and normalized to absolute paths in code

The active config schema now requires this explicit dataset list. Training configs should not rely on glob-based data discovery.

For latent models, `data.coord_space` is also important:

- `zero_to_one`
  - keeps normalized coordinates in `[0, 1]`
- `minus_one_to_one`
  - remaps normalized coordinates into `[-1, 1]`
  - this is the recommended setting for `pointing_cvae`

### Resume behavior

The current config-driven path no longer relies on the notebook-era `prefer_existing_checkpoint` flag.

Replace it with an explicit mode such as:

- `never`
- `latest_matching_version`
- `exact_path`

If resume happens, the code should verify that the embedded resolved config metadata matches the requested config before proceeding.

### W&B behavior

The config should describe W&B explicitly instead of relying on shell-only setup.

Current `wandb` fields are:

- `enabled`
- `project`
- `entity`
- `mode`
- `group`
- `job_type`
- `tags`
- `log_code`
- `log_model_artifact`
- `log_config_artifact`
- `validation_sample_count`
- `log_validation_every_n_epochs`
- `validation_prior_sample_count`
- `validation_prior_temperature`

Current behavior through the shared training path:

- each run initializes a W&B run from the YAML config
- the resolved config is attached to the W&B run config
- key runtime metadata such as split sizes, config hash, and checkpoint paths are added to the W&B summary
- validation previews are uploaded as images and a table on the configured cadence
- `pointing_cvae` runs also log KL, beta, posterior/prior statistics, and validation rows that include posterior mean, prior mean, and prior samples
- the original YAML, resolved config snapshot, and split artifact are uploaded as a W&B artifact
- final and best checkpoints are uploaded as W&B model artifacts when enabled

### Model section

The `model.kind` field selects the instantiated architecture.

Supported values:

- `coord_to_image_unet`
  - deterministic coordinate-conditioned baseline
- `pointing_cvae`
  - posterior encoder over images
  - coordinate-conditioned latent prior
  - reused U-Net decoder fed with concatenated `[coord, z]`

Additional latent-model fields:

- `latent_dim`
- `posterior_base_channels`
- `prior_hidden_dim`

### Training section

Besides the original `device`, `epochs`, `lr`, `amp`, and `resume`, the shared config path now accepts:

- `optimizer`
- `weight_decay`
- `grad_accum_steps`
- `grad_clip_norm`
- `recon_mse_weight`
- `kl_beta_max`
- `kl_beta_warmup_steps`
- `prior_sample_temperature`

## Checkpoint design

### Filename shape

Checkpoint filenames should include the user-provided `VERSION` near the front:

```text
ckpt000123_finger_xy_baseline_v1_coord_to_image_unet_2026-03-13T21-33-38Z.pt
```

Rules:

- keep the monotonic checkpoint ID
- sanitize `VERSION` into a filesystem-safe token
- keep `run_name`
- keep the UTC timestamp

The filename is for humans. Real compatibility should come from checkpoint metadata, not string parsing alone.

### Metadata that is now saved

Current checkpoints embed:

- `checkpoint_format_version`
- full resolved training config
- original config path
- normalized config hash
- git commit hash if available
- dirty-worktree flag if available
- split artifact reference
- model hyperparameters actually instantiated
- training history

For backward compatibility, a flattened compatibility `extra` payload is still saved too.

## Reproducibility requirements

The config system should be opinionated here.

### Frozen split artifacts

A seed alone is not enough once datasets or ordering change.

For real reproducibility:

- persist train/val/test splits to disk
- key them by `VERSION` plus config hash
- store stable sample identifiers such as `(source_dir, frame_index)`
- reuse the frozen split when resuming or rerunning

### Config snapshots

When training starts, write the fully resolved config back out as an artifact, for example:

- next to the checkpoint
- and/or inside `model/artifacts/<VERSION>/`

That protects against later edits to the original YAML file.

### Path normalization

Relative paths in YAML should resolve from the repo root, not from the notebook's working directory.

The loader should store:

- original path string from YAML
- normalized absolute path used at runtime

### Strict validation

The loader should fail fast on:

- unknown keys
- missing required sections
- inconsistent values
- nonexistent required paths
- non-square model image sizes if the model still requires square inputs

This is the main defense against brittle configs.

## Notebook contract

The notebook now should not own training hyperparameters.

Target notebook flow:

1. Choose `CONFIG_PATH`.
2. Load and validate the YAML config.
3. Print a concise summary:
   - config path
   - `VERSION`
   - resolved datasets
   - split sizes
   - model shape
   - resume target
4. Visualize data and config-derived details.
5. Call one config-driven entrypoint such as `run_training_from_config(config)`.

The notebook should be for:

- loading a config
- visualizing what the config means
- running training
- inspecting outputs

The notebook should not duplicate:

- learning rate
- batch size
- image size
- dataset paths
- checkpoint behavior

## Backward compatibility plan

Backward compatibility should be designed in now, even if the first implementation only supports schema `1`.

### Config backward compatibility

- Require `config_schema` in every YAML.
- Keep a migration layer that upgrades older payloads to the latest internal shape.
- Deprecate fields with explicit adapters and warnings, not silent behavior changes.

### Checkpoint backward compatibility

- Continue loading older checkpoints that only have `model_state_dict` plus sparse `extra`.
- When metadata is missing, infer conservatively and surface that it was inferred.
- New checkpoints should always save the full resolved config.

## Current implementation status

Implemented now:

- `configs/`
- strict YAML loading in `model/config.py`
- config-driven training in `model/train.py`
- CLI entrypoint in `scripts/train_from_config.py`
- resolved-config snapshots
- persisted split artifacts
- `VERSION`-prefixed checkpoint filenames
- checkpoint metadata consumed by inference
- config-driven notebook orchestration

Still reasonable follow-up work:

- richer checkpoint selection policies
- best-checkpoint metrics beyond validation loss
- more model kinds than `coord_to_image_unet`
- schema migration helpers once multiple config schema versions exist

## Immediate recommendations for the implementation

- Prefer `PyYAML` plus typed dataclasses if we want minimal new machinery.
- Keep the loader strict and small rather than adding a heavy framework now.
- Make resume behavior explicit and verifiable.
- Save the resolved config in checkpoints from day one.
- Freeze dataset splits instead of relying only on `random_split(seed=...)`.

## Key rule

After this migration, the config file should be the thing you would hand someone else if you wanted them to reproduce the run.

If the run still depends on notebook cell edits, hidden defaults, or manually remembered paths, the refactor is not done.
