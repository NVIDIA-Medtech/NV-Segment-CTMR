# Model Overview

NV-Segment-CTMR is a unified CT and MRI segmentation foundation model. It is based on VISTA3D CT model and extended to both CT and MRI. Please refer to [VISTA3D repo](https://github.com/Project-MONAI/VISTA/tree/main/vista3d) for more information.

We defined 345 classes as in [metadata.json](../configs/metadata.json) and their corresponding dataset in [label_dict.json](../configs/label_dict.json). It shows the label organ name, index, training dataset, modality and evaluation dice score. If a class only comes from CT training dataset, it may not perform well on MRI, but the actual performance will vary case by case.
We support three types of segment everything: "CT_BODY", "MRI_BODY", and "MRI_BRAIN". "CT_BODY" is the previous VISTA3D bundle supported 132 CT classes. "MRI_BODY" shares the same 50 label classes as TotalsegmentatorMR. "MRI_BRAIN" is trained on skull stripped [LUMIR](https://github.com/JHU-MedImage-Reg/LUMIR_L2R) dataset and will segment brain MRI substructures.
Preprocessing is needed. Follow [tutorials](https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_MRI_preprocessing). The exact mapping for those three everything labels can be found in [metadata.json](../configs/metadata.json).

Example segmentations for **CT_BODY** (CT whole-body), **MRI_BRAIN**, and **MRI_BODY** (MRI torso):

![CT_BODY, MRI_BRAIN, and MRI_BODY segmentation examples](ctmr.png)

Note: The predefined segment everything does not cover all labels, user can select more classes as output. Below is a segmentation using the label list from AutoPetAtals. User can extract the label list from each dataset defined in configs/label_mappings.json

![CT_BODY, MRI_BRAIN, and MRI_BODY segmentation examples](ctmr2.png)

Note: For Brain MRI segmentation, the model is able to segment 133 classes across diverse MRI sequences including T1, T2, Flair e.t.c.

## Quick Start

### Installation

```bash
# Create and activate conda environment
conda create -y -n vista3d-nv python=3.11
conda activate vista3d-nv

# Clone repository
git clone https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git
cd NV-Segment-CTMR/NV-Segment-CTMR

# Install dependencies
pip install -r requirements.txt
```

Model weights are prepared automatically during inference. The first run downloads the checkpoint from Hugging Face into the local Hugging Face cache and links it at `models/model.pt`; later runs reuse the cached weights.

## Automatic Segmentation (support multi-gpu batch processing)

## Single image inference to segment everything (automatic)

The output will be saved to `{output_dir}/s0289/s0289_{output_postfix}{output_ext}`. By default the everything will be "CT_BODY". Add "MRI_BODY" to segment the MRI body classes.

```bash
# Make sure conda environment is activated
conda activate vista3d-nv

# Automatic Segment everything. It requires a modality key. We allow "CT_BODY", "MRI_BODY", and "MRI_BRAIN". For brain, we require preprocessing.
python -m monai.bundle run --config_file configs/inference.json --input_dict "{'image':'example/s0289.nii.gz'}" --modality MRI_BODY
```

## Single image inference to segment specific class (automatic)

The detailed automatic segmentation class index can be found [here](../configs/label_dict.json).

```bash
# Automatic Segment specific class
python -m monai.bundle run --config_file configs/inference.json --input_dict "{'image':'example/s0289.nii.gz','label_prompt':[3]}"
```

## Batch inference with multiGPU support (automatic)

The `configs/batch_inference.json` defines the batch inference, you can:

1. Segment all NIfTI files within a folder and subfolders

    - `configs/batch_inference.json` builds `input_list` with `scripts/batch_inference_utils.build_input_list()`:
    - Recursively discovers `**/*.nii.gz` under `--input_dir`.
    - **Resume (default):** with `batch_resume_skip_existing: true` in `batch_inference.json`, only volumes whose expected output is **missing or empty** under `--output_dir` are queued (same layout as `SaveImaged`). Re-run the **same** command to finish leftovers. Set `batch_resume_skip_existing` to false to segment every discovered file again.
    - **Discovery filters:** edit `batch_skip_dir_names` (exact parent folder name) and/or `batch_skip_dir_prefixes` (parent folder name starts with…) in `configs/batch_inference.json` (JSON arrays of strings).
    - **Optional keys** in `configs/batch_inference.json` (defaults in the file): `batch_skip_dir_names`, `batch_skip_dir_prefixes`, `batch_resume_skip_existing`, `batch_use_input_list_cache`, `batch_cache_wait_sec`.
    - **`batch_use_input_list_cache`:** With `torchrun` (multi-process), only rank 0 walks the tree to build `input_list` and writes a small JSON cache under the system temp directory; other ranks read that file so you do not repeat a huge filesystem scan on every GPU. Set to `false` if you want every rank to compute the list itself (simpler, slower on large cohorts). Single-process runs are unaffected in practice.
    - **`batch_cache_wait_sec`:** When `batch_use_input_list_cache` is `true`, non-zero ranks wait up to this many seconds for rank 0’s cache file. Increase if rank 0’s scan is slow; decrease only if the list is always built quickly.
    - **Which classes to segment:** edit **`everything_labels`** in **`configs/inference.json`** (and `modality` / `--modality` as needed). See `configs/label_dict.json` and `docs/inference.md`.
    - If **resume** leaves nothing to run (all outputs already present), the run **exits successfully** with `[nvseg] batch: nothing to run (resume); ok` (avoids a zero-length dataloader / `DistributedSampler` failure). If **no** `*.nii.gz` files are discovered under `input_dir`, you get a short `[nvseg] batch: no *.nii.gz…` error.
    - Rank 0 logs: `[nvseg] batch resume (skip existing outputs): N volume(s) (...)`.
    - **Multi-GPU:** `--nproc_per_node` must be ≤ the number of volumes in `input_list` after filtering.
    - **Outputs:** With `data_root_dir` and `separate_folder: true`, `input_dir/patient1/mri/scan.nii.gz` → `output_dir/patient1/mri/scan/scan_trans.nii.gz`. If `models/model.pt` is missing, inference prepares it automatically from Hugging Face.
    - Advanced: edit `should_skip_path_by_parent_rules()` in `scripts/batch_inference_utils.py` for custom path rules.

2. Segment based on a filelist.txt file, you can change the `input_list` in `configs/batch_inference.json`

```json
  "input_list": "$sorted([os.path.abspath(line.strip()) for line in open('/absolute/path/to/filelist.txt') if line.strip() and not line.strip().startswith('#')])",
```

### Single-GPU Batch Inference

```bash
# Make sure conda environment is activated
conda activate vista3d-nv

# Segment MRI_BODY within example folder
python -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json']" --input_dir="example/" --output_dir="example/" --modality MRI_BODY
```

### Multi-GPU batch inference (cohorts, resume, optional folder filters)

```bash
conda activate vista3d-nv

# Example: multi-GPU batch (same command for first run or resume)
torchrun --nproc_per_node=2 --nnodes=1 -m monai.bundle run \
  --config_file="['configs/inference.json', 'configs/batch_inference.json', 'configs/mgpu_inference.json']" \
  --input_dir="example/" \
  --output_dir="example/" \
  --modality MRI_BODY
```

```text
Note: if using the finetuned checkpoint and the finetuning label_mapping mapped to global index "2, 20, 21", remove the `subclass` dict from inference.json since those values defined in `subclass` will trigger the wrong subclass segmentation.
```

## Brain MRI segmentation (any MRI sequence)

### Using the Brain Segmentation Script

The script automates skull stripping (SynthStrip via `brain_t1_preprocess/synthstrip-docker`), affine alignment to the LUMIR template, MONAI bundle inference, and reverting the mask to the original image space. Temporary files are removed after each case unless you pass `--keep-temp`. It is modified from [MIR tutorials](https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_MRI_preprocessing).

#### Single file

Output path: `{output_dir}/{basename}_trans.nii.gz` (default `output_dir` is `./eval`).

```bash
conda activate vista3d-nv

./brain_t1_preprocess/run_brain_segmentation.sh --input example/brain_t1.nii.gz
./brain_t1_preprocess/run_brain_segmentation.sh --input example/brain_t1.nii.gz --output_dir results/
# keep temperary files for skull stripping and registration
./brain_t1_preprocess/run_brain_segmentation.sh --input example/brain_t1.nii.gz --keep-temp
# if you cannot perform skull stripping with synthstrip-docker, e.g. on a remote cluster in docker env, you can skull strip the data first then run segmentation with skull strip skipped.
./brain_t1_preprocess/run_brain_segmentation.sh --input example/brain_t1.nii.gz --no-skullstrip
```

#### Folder batch (`--input_folder`)

Only `*.nii` / `*.nii.gz` files **directly inside** the given folder are processed (`find` with `-maxdepth 1`; no subfolders). Each output is written as `{output_dir}/{basename}_trans.nii.gz` (flat layout).

```bash
./brain_t1_preprocess/run_brain_segmentation.sh --input_folder example/ --output_dir results/
./brain_t1_preprocess/run_brain_segmentation.sh --input_folder example/ --output_dir results/ --keep-temp
```

#### File list (`--file_list` + `--root_path`)

Lines in the list are paths relative to `root_path` (comments and empty lines allowed). Outputs mirror that relative layout under `output_dir`, with `_seg` before the extension (e.g. `root_path/sub/scan.nii.gz` → `output_dir/sub/scan_seg.nii.gz`). By default, existing outputs are skipped (resume-friendly). Optional: split the sorted list across jobs with `--num_partitions N` and `--partition M` (1-based).

```bash
./brain_t1_preprocess/run_brain_segmentation.sh \
  --file_list file_list.txt --root_path /path/to/root --output_dir /path/to/output
./brain_t1_preprocess/run_brain_segmentation.sh \
  --file_list file_list.txt --root_path /path/to/root --output_dir /path/to/output --no-skullstrip
# split into multiple partitions to submit multiple jobs
./brain_t1_preprocess/run_brain_segmentation.sh \
  --file_list file_list.txt --root_path /path/to/root --output_dir /path/to/output \
  --num_partitions 10 --partition 3
```

#### Script options

- `--input FILE`: Single NIfTI to segment (mutually exclusive with `--input_folder` / `--file_list`).
- `--input_folder FOLDER`: Batch only the NIfTI files in that folder (not recursive).
- `--file_list FILE` and `--root_path PATH`: Batch from a text list (requires both).
- `--output_dir DIR`: Output directory (default `./eval`).
- `--keep-temp`: Keep per-case temporary preprocessing files (default: delete after success).
- `--no-skullstrip`: Skip SynthStrip; preprocessing starts from the original image.
- `--modality MODALITY`: `MRI_BRAIN` (default), `MRI_BODY`, or `CT_BODY`.
- `--no-skip`: Recompute even when the expected output already exists (**file list mode only**).
- `--num_partitions N`, `--partition M`: Deterministic shards of the sorted file list (**file list mode only**).
- `-h`, `--help`: Print usage.

**Note:** Skull stripping calls `brain_t1_preprocess/synthstrip-docker` (Docker must be available unless you use `--no-skullstrip`). In file list mode, failed or timed-out cases are appended to a timestamped log under the output directory.

### Manual Processing (Advanced)

If you need more control over individual steps, you can run them manually:

```bash
# Make sure conda environment is activated
conda activate vista3d-nv

# Set variables
file=brain_t1
input=example/$file.nii.gz
skull_stripped=example/${file}_skull_stripped.nii.gz
preprocess_tmp=example/${file}_p.nii.gz
preprocess_meta=example/${file}_p.meta.json

# Step 1: Skull stripping with SynthStrip
./brain_t1_preprocess/synthstrip-docker -i $input -o $skull_stripped

# Step 2: Affine align to the LUMIR template
python brain_t1_preprocess/preprocess.py $skull_stripped brain_t1_preprocess/LUMIR_template.nii.gz $preprocess_tmp --save-preprocess $preprocess_meta

# Step 3: Segment the brain
python -m monai.bundle run --config_file configs/inference.json --input_dict "{'image':'$preprocess_tmp'}" --modality MRI_BRAIN

# Step 4: Revert the segmentation back to original space
# Note: Adjust paths based on actual output location from step 3
python brain_t1_preprocess/revert_preprocess.py $preprocess_tmp --out ${preprocess_tmp}.revert.nii.gz --mask eval/${file}_p/${file}_p_trans.nii.gz --mask-out eval/${file}_trans.nii.gz --meta $preprocess_meta
```

## Execute inference with the TensorRT model

```bash
# Make sure conda environment is activated
conda activate vista3d-nv

python -m monai.bundle run --config_file "['configs/inference.json', 'configs/inference_trt.json']"
```

## Continual learning / Finetuning

See the [finetuning tutorial](finetune.md) for the standard VISTA3D continual-learning workflow.

For the self-contained fixed-channel training and inference workflow, see
[Softmax Finetuning](softmax_finetune.md).

For complicated finetuning, users can build custom pipelines by reusing the model
and checkpoint:

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

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

### Model Weights License

The model weights included in this project are licensed under the Non-Commercial

[NCLS v1 License](https://developer.download.nvidia.com/licenses/NVIDIA-OneWay-Noncommercial-License-22Mar2022.pdf?t=eyJscyI6InJlZiIsImxzZCI6IlJFRi1naXRodWIuY29tL252aWRpYS1ob2xvc2NhbiJ9)
