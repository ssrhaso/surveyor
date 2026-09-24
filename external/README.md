# External code and data

Nothing here is vendored. The sources below are public; clone or download them at the versions given.

## Code

| Used by | Repository | Commit |
|---|---|---|
| `surveyor/gcidm_official.py` (the authors' GC-IDM: `idm` package, `train_idm.py extract` / `train`) | https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models | `48c45b1cb2b34dd2c1c61d222c8309de567fde55` |
| `surveyor/leflow_executor.py` (LeFlow: `latent_planner.py`, `train_latent_planner.py`) | https://github.com/hsiangwei0903/LeFlow | `f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5`, plus the change below |

The code expects each clone under `external/` with the repository's own name:

```
git clone https://github.com/hdnndh/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models \
    external/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models
git -C external/Latent-Geometry-Beyond-Search-Amortizing-Planning-in-World-Models checkout 48c45b1cb2b34dd2c1c61d222c8309de567fde55
git clone https://github.com/hsiangwei0903/LeFlow external/LeFlow
git -C external/LeFlow checkout f1fe192e41ec6f20de25cd1054ede40a8cbdcfa5
```

One change to LeFlow, for compatibility with stable-pretraining 0.1.8, whose `Resize` transform wrapper has no
`.transform` attribute. The datasets are rendered at `img_size` already, so the resize is a no-op and is dropped
(`external/LeFlow/utils.py`, `get_img_preprocessor`):

```diff
     to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
-    resize = dt.transforms.Resize(img_size, source=source, target=target)
-    return dt.transforms.Compose(to_image, resize)
+    return dt.transforms.Compose(to_image)
```

## Python packages

`requirements.txt` lists the stack. The two packages the planners and datasets come from were pinned at:

```
pip install stable-worldmodel==0.1.1 stable-pretraining==0.1.8
```

The runs used Python 3.11 with torch 2.5.1 and torchvision 0.20.1 (CUDA 12.1 wheels), transformers 4.57.1, numpy 2.4.6,
h5py 3.16.0, hdf5plugin 7.0.0, mujoco 3.10.0, dm_control 1.0.43, ogbench 1.2.1, gymnasium 1.3.0, lightning 2.6.6 and
hydra-core 1.3.6.

## Datasets and pretrained models

All from Hugging Face (`https://huggingface.co/datasets/<id>/resolve/main/<file>` for the datasets). Put the h5 files
in one directory and point `DATA` at it (see `reproduce/README.md`).

| Environment | Dataset (id, file) | h5 in `DATA` | LeWM encoder and predictor |
|---|---|---|---|
| PushT | `quentinll/lewm-pusht`, `pusht_expert_train.h5.zst` | `pusht_expert_train.h5` | `quentinll/lewm-pusht` (`--source pretrained`, the default `--encoder-id`) |
| Reacher | `quentinll/lewm-reacher`, `reacher.tar.zst` | `reacher.h5` | `quentinll/lewm-reacher`: `config.json` and `weights.pt` in `encoder_reacher/` (`--source local --local-dir encoder_reacher`) |
| Cube | `quentinll/lewm-cube`, `cube_single_expert.tar.zst` | `cube_single_expert.h5` | `quentinll/lewm-cube` (`--encoder-id quentinll/lewm-cube`) |
| Two-Room | `quentinll/lewm-tworooms`, `tworoom.tar.zst` | `tworoom.h5` (run `reproduce/data_prep/tworoom_add_state.py` on it once) | `quentinll/lewm-tworooms` (`--encoder-id quentinll/lewm-tworooms`) |

The Two-Room accept rule reads DINOv2 ViT-S/14 features, loaded with `torch.hub.load("facebookresearch/dinov2",
"dinov2_vits14")` (https://github.com/facebookresearch/dinov2).
