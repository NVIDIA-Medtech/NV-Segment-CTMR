# Model Overview

NV-Segment-CT is a copy from the VISTA3D monai model zoo. This is the Vista3D model fintuning/evaluation/inference pipeline. VISTA3D is trained using over 20 partial datasets with more complicated pipeline. To avoid confusion, we will only provide finetuning/continual learning APIs for users to finetune on their
own datasets. To reproduce the paper results, please refer to [VISTA3D repo](https://github.com/Project-MONAI/VISTA/tree/main/vista3d).

## Quick Start

### Installation

```bash
# use the same conda env as this repo
conda create -y -n vista3d-nv python=3.11
conda activate vista3d-nv
git clone https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git
cd NV-Segment-CTMR/NV-Segment-CT;
pip install -r requirements.txt;
```

Model weights are prepared automatically during inference. The first run downloads the checkpoint from Hugging Face into the local Hugging Face cache and links it at `models/model.pt`; later runs reuse the cached weights.

## 1.1 **NV-Segment-CT** [[Github]](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/main/NV-Segment-CT) [[Huggingface]](https://huggingface.co/nvidia/NV-Segment-CT)

### Automatic Segmentation (support multi-gpu batch processing)

[class definition](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/blob/main/NV-Segment-CT/configs/label_dict.json)

#### Single image inference to segment everything (automatic)

The output will be saved to `{output_dir}/spleen_03/spleen_03_{output_postfix}{output_ext}`.

```bash
# Make sure conda environment is activated
conda activate vista3d-nv
cd NV-Segment-CT

# Automatic Segment everything
python -m monai.bundle run --config_file configs/inference.json --input_dict "{'image':'example/spleen_03.nii.gz'}"
```

#### Single image inference to segment specific class (automatic)

The detailed automatic segmentation class index can be found [here](../configs/label_dict.json).

```bash
# Automatic Segment specific class
python -m monai.bundle run --config_file configs/inference.json --input_dict "{'image':'example/spleen_03.nii.gz','label_prompt':[3]}"
```

#### Batch inference with multiGPU support (automatic)

The `configs/batch_inference.json` defines the batch inference, you can:

1. Segment all NIfTI files within a folder and subfolders

    - `configs/batch_inference.json` builds `input_list` with `scripts/batch_inference_utils.build_input_list()`:
    - Recursively discovers `**/*.nii.gz` under `--input_dir`.
    - **Resume (default):** with `batch_resume_skip_existing: true` in `batch_inference.json`, only volumes whose expected output is **missing or empty** under `--output_dir` are queued (same layout as `SaveImaged`). Re-run the **same** command to finish leftovers. Set `batch_resume_skip_existing` to false to segment every discovered file again.
    - **Discovery filters:** edit `batch_skip_dir_names` (exact parent folder name) and/or `batch_skip_dir_prefixes` (parent folder name starts with...) in `configs/batch_inference.json` (JSON arrays of strings).
    - **Optional keys** in `configs/batch_inference.json` (defaults in the file): `batch_skip_dir_names`, `batch_skip_dir_prefixes`, `batch_resume_skip_existing`, `batch_use_input_list_cache`, `batch_cache_wait_sec`.
    - **`batch_use_input_list_cache`:** With `torchrun` (multi-process), only rank 0 walks the tree to build `input_list` and writes a small JSON cache under the system temp directory; other ranks read that file so you do not repeat a huge filesystem scan on every GPU. Set to `false` if you want every rank to compute the list itself (simpler, slower on large cohorts). Single-process runs are unaffected in practice.
    - **`batch_cache_wait_sec`:** When `batch_use_input_list_cache` is `true`, non-zero ranks wait up to this many seconds for rank 0's cache file. Increase if rank 0's scan is slow; decrease only if the list is always built quickly.
    - **Which classes to segment:** edit **`everything_labels`** in **`configs/inference.json`**. See `configs/label_dict.json` and `docs/inference.md`.
    - If **resume** leaves nothing to run (all outputs already present), the run **exits successfully** with `[nvseg] batch: nothing to run (resume); ok` (avoids a zero-length dataloader / `DistributedSampler` failure). If **no** `*.nii.gz` files are discovered under `input_dir`, you get a short `[nvseg] batch: no *.nii.gz...` error.
    - Rank 0 logs: `[nvseg] batch resume (skip existing outputs): N volume(s) (...)`.
    - **Multi-GPU:** `--nproc_per_node` must be less than or equal to the number of volumes in `input_list` after filtering.
    - **Outputs:** With `data_root_dir` and `separate_folder: true`, `input_dir/patient1/scan.nii.gz` -> `output_dir/patient1/scan/scan_trans.nii.gz`. If `models/model.pt` is missing, inference prepares it automatically from Hugging Face.
    - Advanced: edit `should_skip_path_by_parent_rules()` in `scripts/batch_inference_utils.py` for custom path rules.

2. Segment based on a filelist.txt file, you can change the `input_list` in `configs/batch_inference.json`

```json
  "input_list": "$sorted([os.path.abspath(line.strip()) for line in open('/absolute/path/to/filelist.txt') if line.strip() and not line.strip().startswith('#')])",
```

##### Single-GPU Batch Inference

```bash
# Make sure conda environment is activated
conda activate vista3d-nv
cd NV-Segment-CT

python -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json']" --input_dir="example/" --output_dir="example/"
```

##### Multi-GPU batch inference (cohorts, resume, optional folder filters)

```bash
conda activate vista3d-nv
cd NV-Segment-CT

# Example: multi-GPU batch (same command for first run or resume)
torchrun --nproc_per_node=2 --nnodes=1 -m monai.bundle run \
  --config_file="['configs/inference.json', 'configs/batch_inference.json', 'configs/mgpu_inference.json']" \
  --input_dir="example/" \
  --output_dir="example/"
```

```text
Note: if using a finetuned checkpoint whose label_mapping maps custom labels to global indexes "2, 20, 21", remove the `subclass` dict from inference.json since those values defined in `subclass` will trigger the wrong subclass segmentation.
```

### Interactive segmentation

```bash
# Points must be three dimensional (x,y,z) in the shape of [[x,y,z],...,[x,y,z]]. Point labels can only be -1(ignore), 0(negative), 1(positive) and 2(negative for special overlaped class like tumor), 3(positive for special class). Only supporting 1 class per inference. The output 255 represents NaN value which means not processed region. If you provide label_prompt at the same time, the results will be auto + interactive refinement.
cd NV-Segment-CT
python -m monai.bundle run --config_file configs/inference.json --input_dict "{'image':'example/spleen_03.nii.gz','points':[[128,128,16], [100,100,16]],'point_labels':[1, 0]}"
```

**NOTE** MONAI bundle accepts multiple json config files and input arguments. The latter configs/arguments will overide the previous configs/arguments if they have overlapping keys.

## Configuration details and interactive segmentation

For inference, VISTA3d bundle requires at least one prompt for segmentation. It supports label prompt, which is the index of the class for automatic segmentation.
It also supports point click prompts for binary interactive segmentation. User can provide both prompts at the same time. Please refer to [this](inference.md).

## Execute inference with the TensorRT model

```bash
python -m monai.bundle run --config_file "['configs/inference.json', 'configs/inference_trt.json']"
```

For more details, please refer to [this](inference.md).

## Continual learning / Finetuning

For conventional fixed-channel training and inference, see
[Softmax Finetuning](softmax_finetune.md).

We provide the standard VISTA3D finetuning tutorial in [details](finetune.md).
For complicated finetuning, we suggest users to do vibe coding to generate finetuning pipelines by simply reuse the model and checkpoint

```python
from monai.networks.nets.vista3d import vista3d132
vista3d132.load_state_dict(pretrained_ckpt, strict=True)
```

## References

- He, Yufan, et al. "VISTA3D: A unified segmentation foundation model for 3D medical imaging." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025. <https://openaccess.thecvf.com/content/CVPR2025/html/He_VISTA3D_A_Unified_Segmentation_Foundation_Model_For_3D_Medical_Imaging_CVPR_2025_paper.html>

## License

### Code License

This project includes code licensed under the Apache License 2.0.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

### Model Weights License

THe model weights license is under commercial friendly

[NVIDIA open model license](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
