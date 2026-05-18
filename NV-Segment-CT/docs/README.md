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

Model weights are prepared automatically during inference. The first run downloads the checkpoint from Hugging Face into the local Hugging Face cache and links it at `models/model.pt`;

## 1.1 **NV-Segment-CT** [[Github]](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/tree/main/NV-Segment-CT) [[Huggingface]](https://huggingface.co/nvidia/NV-Segment-CT)

### Automatic Segmentation (support multi-gpu batch processing)

[class definition](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR/blob/main/NV-Segment-CTMR/configs/label_dict.json)

```bash
# CT sementation
cd NV-Segment-CT
# Automatic Segment everything
python -m monai.bundle run --config_file configs/inference.json --input_dict "{'image':'example/spleen_03.nii.gz'}"
# Automatic Segment specific class
python -m monai.bundle run --config_file configs/inference.json --input_dict "{'image':'example/spleen_03.nii.gz','label_prompt':[3]}"
# Automatic Batch segmentation for the whole folder
python -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json']" --input_dir="example/" --output_dir="example/"
# Automatic Batch segmentation for the whole folder with multi-gpu support. mgpu_inference.json is below. change nproc_per_node to your GPU number.
torchrun --nproc_per_node=2 --nnodes=1 -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json', 'configs/mgpu_inference.json']" --input_dir="example/" --output_dir="example/"
```

Note: For more details about batch processing, please refer to NV-Segment-CTMR readme.md

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

We provide predefined finetuning tutorial in [details](finetune.md).
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
