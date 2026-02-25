# Model Overview

NV-Segment-CTMR is a unified CT and MRI segmentation foundation model. It is based on VISTA3D CT model and extended to both CT and MRI. Please refer to [VISTA3D repo](https://github.com/Project-MONAI/VISTA/tree/main/vista3d) for more information.

## Performance on held-out test set

![Benchmark CT](./benchmarkct.png) ![Benchmark MR](./benchmarkmr.png)

## Quick Start

### Installation

```bash
# Create and activate conda environment
conda create -y -n vista3d-nv python=3.9
conda activate vista3d-nv

# Clone repository
git clone https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git
cd NV-Segment-CTMR/NV-Segment-CTMR

# Install dependencies
pip install -r requirements.txt

# Create models directory and download pretrained model
cd ..
mkdir -p NV-Segment-CTMR/models
wget -O NV-Segment-CTMR/models/model.pt https://huggingface.co/nvidia/NV-Segment-CTMR/resolve/main/vista3d_pretrained_model/model.pt
```

## Automatic Segmentation (support multi-gpu batch processing)

We defined 345 classes as in [label_dict.json](../configs/label_dict.json). It shows the label organ name, index, training dataset, modality and evaluation dice score. If a class only comes from CT training dataset, it may not perform well on MRI, but the actual performance will vary case by case. We support three types of segment everything: "CT_BODY", "MRI_BODY", and "MRI_BRAIN". "CT_BODY" is the previous VISTA3D bundle supported 132 CT classes. "MRI_BODY" shares the same 50 label classes as TotalsegmentatorMR. "MRI_BRAIN" is trained on skull stripped [LUMIR](https://github.com/JHU-MedImage-Reg/LUMIR_L2R) dataset and will segment brain MRI substructures.
Preprocessing is needed. Follow [tutorials](https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_MRI_preprocessing). The exact mapping for those three everything labels can be found in [metadata.json](../configs/metadata.json).

## Single image inference to segment everything (automatic)

The output will be saved to `output_dir/s0289/s0289_{output_postfix}{output_ext}`. By default the everything will be "CT_BODY". Add "MRI_BODY" to segment the MRI body classes.

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

## Batch inference with multiGPU support for segmenting everything (automatic)

### Single-GPU Batch Inference

```bash
# Make sure conda environment is activated
conda activate vista3d-nv

# Segment MRI_BODY within example folder
python -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json']" --input_dir="example/" --output_dir="example/" --modality MRI_BODY
```

### Multi-GPU Batch Inference

**Important**: Always activate your conda environment before running `torchrun`. If you don't, you may encounter `ModuleNotFoundError` because `torchrun` will use the system Python instead of your conda environment's Python.

```bash
# Activate conda environment first (CRITICAL!)
conda activate vista3d-nv

# Automatic Batch segmentation for the whole folder with multi-gpu support
# Change --nproc_per_node to match your number of GPUs
torchrun --nproc_per_node=2 --nnodes=1 -m monai.bundle run --config_file="['configs/inference.json', 'configs/batch_inference.json', 'configs/mgpu_inference.json']" --input_dir="example/" --output_dir="example/"
```

`configs/batch_inference.json` by default runs the segment everything workflow (classes defined by `everything_labels`) on all (`*.nii.gz`) files in `input_dir`.
This default is overridable by changing the input folder `input_dir`, or the input image name suffix `input_suffix`, or directly setting the list of filenames `input_list`.

```text
Note: if using the finetuned checkpoint and the finetuning label_mapping mapped to global index "2, 20, 21", remove the `subclass` dict from inference.json since those values defined in `subclass` will trigger the wrong subclass segmentation.
```

## Brain MRI segmentation

For brain MRI segmentation, we only support T1 and require preprocessing. We provide a convenient bash script that handles all preprocessing steps automatically.

### Using the Brain Segmentation Script

The script `brain_t1_preprocess/run_brain_segmentation.sh` automates the entire pipeline: skull stripping, preprocessing, segmentation, and reverting results back to original space. It also handles temporary file cleanup automatically. It is modified from [MIR tutorials](https://github.com/junyuchen245/MIR/tree/main/tutorials/brain_MRI_preprocessing).

#### Single File Processing

```bash
# Process a single brain MRI file
./brain_t1_preprocess/run_brain_segmentation.sh --input example/brain_t1.nii.gz

# Specify custom output directory
./brain_t1_preprocess/run_brain_segmentation.sh --input example/brain_t1.nii.gz --output_dir results/

# Keep temporary files for debugging
./brain_t1_preprocess/run_brain_segmentation.sh --input example/brain_t1.nii.gz --keep-temp
```

#### Batch Processing

```bash
# Process all NIfTI files in a folder
./brain_t1_preprocess/run_brain_segmentation.sh --input_folder example/ --output_dir results/

# Batch processing with temporary files kept
./brain_t1_preprocess/run_brain_segmentation.sh --input_folder example/ --keep-temp
```

#### Script Options

- `--input FILE`: Single NIfTI file to segment
- `--input_folder FOLDER`: Folder containing NIfTI files (batch mode)
- `--output_dir DIR`: Output directory (default: `./eval`)
- `--keep-temp`: Keep temporary preprocessing files (default: false, files are cleaned up automatically)
- `--modality MODALITY`: Segmentation modality: `MRI_BRAIN` (default), `MRI_BODY`, `CT_BODY`
- `--conda-env ENV`: Conda environment name (default: `vista3d-nv`)
- `-h, --help`: Show help message

**Note**: The script automatically activates the conda environment and handles all temporary file management. By default, temporary files are cleaned up after processing. Use `--keep-temp` if you need to inspect intermediate results.

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

For more details, please refer to [this](inference.md).

## Continual learning / Finetuning

### Step1: Generate Data json file

Users need to provide a json data split for continuous learning (`configs/msd_task09_spleen_folds.json` from the [MSD](http://medicaldecathlon.com/) is provided as an example). The data split should meet the following format ('testing' labels are optional):

```json
{
    "training": [
        {"image": "img0001.nii.gz", "label": "label0001.nii.gz", "fold": 0},
        {"image": "img0002.nii.gz", "label": "label0002.nii.gz", "fold": 2},
        ...
     ],
    "testing": [
        {"image": "img0003.nii.gz", "label": "label0003.nii.gz"},
        {"image": "img0004.nii.gz", "label": "label0004.nii.gz"},
        ...
   ]
}
```

Example code for 5 fold cross-validation generation can be found [here](data.md)

```text
Note the data is not the absolute path to the image and label file. The actual image file will be `os.path.join(dataset_dir, data["training"][item]["image"])`, where `dataset_dir` is defined in `configs/train_continual.json`. Also 5-fold cross-validation is not required! `fold=0` is defined in train.json, which means any data item with fold==0 will be used as validation and other fold will be used for training. So if you only have train/val split, you can manually set validation data with "fold": 0 in its datalist and the other to be training by setting "fold" to any number other than 0.
```

### Step2: Changing hyperparameters

For continual learning, user can change `configs/train_continual.json`. More advanced users can change configurations in `configs/train.json`.  Most hyperparameters are straighforward and user can tell based on their names. The users must manually change the following keys in `configs/train_continual.json`.

#### 1. `label_mappings`

```json
    "label_mappings": {
        "default": [
            [
                index_1_in_user_data, # e.g. 1
                mapped_index_1, # e.g. 1
            ],
            [
                index_2_in_user_data, # e.g. 2
                mapped_index_2, # e.g. 2
            ], ...,
            [
                index_last_in_user_data, # e.g. N
                mapped_index_N, # e.g. N
            ]
        ]
    },
```

`index_1_in_user_data`,...,`index_N_in_user_data` is the class index value in the groundtruth that user tries to segment. `mapped_index_1`,...,`mapped_index_N` is the mapped index value that the bundle will output. You can make these two the same for finetuning, but we suggest finding the semantic relevant mappings from our unified [global label index](../configs/metadata.json). For example, "Spleen" in MSD09 groundtruth label is represented by 1, but "Spleen" is 3 in `docs/labels.json`. So by defining label mapping `[[1, 3]]`, VISTA3D can segment "Spleen" using its pretrained weights out-of-the-box,
and can speed up the finetuning convergence speed.
If you cannot find a relevant semantic label for your class, just use any value < `num_classes` defined in train_continue.json.
For more details about this label_mapping, please read [this](finetune.md).

#### 2.  `data_list_file_path` and `dataset_dir`

Change `data_list_file_path` to the absolute path of your data json split. Change `dataset_dir` to the root folder that combines with the relative path in the data json split.

#### 3. Optional hyperparameters and details are [here](finetune.md)

Hyperparameter finetuning is important and varies from task to task.

## Step3: Run finetuning

The hyperparameters in `configs/train_continual.json` will overwrite ones in `configs/train.json`. Configs in the back will overide the previous ones if they have the same key.

Single-GPU:

```bash
# Make sure conda environment is activated
conda activate vista3d-nv

python -m monai.bundle run \
    --config_file="['configs/train.json','configs/train_continual.json']"
```

Multi-GPU:

```bash
# Activate conda environment first (CRITICAL!)
conda activate vista3d-nv

# Change --nproc_per_node to match your number of GPUs
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run \
    --config_file="['configs/train.json','configs/train_continual.json','configs/multi_gpu_train.json']"
```

### MLFlow Visualization

MLFlow is enabled by default (defined in train.json, use_mlflow) and the data is stored in the `mlruns/` folder under the bundle's root directory. To launch the MLflow UI and track your experiment data, follow these steps:

1. Open a terminal and navigate to the root directory of your bundle where the `mlruns/` folder is located.

2. Execute the following command to start the MLflow server. This will make the MLflow UI accessible.

```bash
mlflow ui
```

## Evaluation

Evaluation can be used to calculate dice scores for the model or a finetuned model. Change the `ckpt_path` to the checkpoint you wish to evaluate. The dice score is calculated on the original image spacing using `invertd`, while the dice score during finetuning is calculated on resampled space.

```text
NOTE: Evaluation does not support point evaluation.`"validate#evaluator#hyper_kwargs#val_head` is always set to `auto`.
```

Single-GPU:

```bash
# Make sure conda environment is activated
conda activate vista3d-nv

python -m monai.bundle run \
    --config_file="['configs/train.json','configs/train_continual.json','configs/evaluate.json']"
```

Multi-GPU:

```bash
# Activate conda environment first (CRITICAL!)
conda activate vista3d-nv

# Change --nproc_per_node to match your number of GPUs
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run \
    --config_file="['configs/train.json','configs/train_continual.json','configs/evaluate.json','configs/mgpu_evaluate.json']"
```

### Other explanatory items

The `label_mapping` in `evaluation.json` does not include `0` because the postprocessing step performs argmax (`VistaPostTransformd`), and a `0` prediction would negatively impact performance. In continuous learning, however, `0` is included for validation because no argmax is performed, and validation is done channel-wise (include_background=False). Additionally, `Relabeld` in `postprocessing` is required to map `label` and `pred` back to sequential indexes like `0, 1, 2, 3, 4` for dice calculation, as they are not in one-hot format. Evaluation does not support `point`, but finetuning does, as it does not perform argmax.

## FAQ

### TroubleShoot for Out-of-Memory

- Changing `patch_size` to a smaller value such as `"patch_size": [96, 96, 96]` would reduce the training/inference memory footprint.
- Changing `train_dataset_cache_rate` and `val_dataset_cache_rate` to a smaller value like `0.1` can solve the out-of-cpu memory issue when using huge finetuning dataset.
- Set `"postprocessing#transforms#0#_disabled_": false` to move the postprocessing to cpu to reduce the GPU memory footprint.

### Multi-channel input

- Change `input_channels` in `train.json` to your desired channel number
- Data split json can be a single multi-channel image or can be a list of single channeled images. Those images must have the same spatial shape and aligned/registered.

```json
        {
            "image": ["modality1.nii.gz", "modality2.nii.gz", "modality3.nii.gz"]
            "label": "label.nii.gz"
        },
```

### Wrong inference results from finetuned checkpoint

- Make sure you removed the `subclass` dictionary from inference.json if you ever mapped local index to [2,20,21]
- Make sure `0` is not included in your inference prompt for automatic segmentation.

## References

- Antonelli, M., Reinke, A., Bakas, S. et al. The Medical Segmentation Decathlon. Nat Commun 13, 4128 (2022). <https://doi.org/10.1038/s41467-022-30695-9>

- VISTA3D: Versatile Imaging SegmenTation and Annotation model for 3D Computed Tomography. arxiv (2024) <https://arxiv.org/abs/2406.05285>

## License

### Code License

This project includes code licensed under the Apache License 2.0.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

### Model Weights License

The model weights included in this project are licensed under the NCLS v1 License.

Both licenses' full texts have been combined into a single `LICENSE` file. Please refer to this `LICENSE` file for more details about the terms and conditions of both licenses.

For MRI CT joint model. The license is non-commercial and needs future discussion.
