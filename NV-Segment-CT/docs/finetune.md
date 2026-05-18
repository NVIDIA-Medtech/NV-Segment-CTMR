# Finetune configurations

## Step1: Generate Data json file

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

## Step2: Changing hyperparameters

For continual learning, user can change `configs/train_continual.json`. More advanced users can change configurations in `configs/train.json`.  Most hyperparameters are straighforward and user can tell based on their names. The users must manually change the following keys in `configs/train_continual.json`.

### 1. `label_mappings`

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

`index_1_in_user_data`,...,`index_N_in_user_data` is the class index value in the groundtruth that user tries to segment. `mapped_index_1`,...,`mapped_index_N` is the mapped index value that the bundle will output. You can make these two the same for finetuning, but we suggest finding the semantic relevant mappings from our unified [global label index](../configs/metadata.json). For example, "Spleen" in MSD09 groundtruth label is represented by 1, but "Spleen" is 3 in `docs/labels.json`. So by defining label mapping `[[1, 3]]`, VISTA3D can segment "Spleen" using its pretrained weights out-of-the-box, and can speed up the finetuning convergence speed.
If you cannot find a relevant semantic label for your class, just use any value < `num_classes` defined in train_continue.json.
For more details about this label_mapping, please read [this](finetune.md).

### 2.  `data_list_file_path` and `dataset_dir`

Change `data_list_file_path` to the absolute path of your data json split. Change `dataset_dir` to the root folder that combines with the relative path in the data json split.

### 3. Optional hyperparameters and details are [here](finetune.md)

Hyperparameteers finetuning is important and varies from task to task.

## Step3: Run finetuning

The hyperparameters in `configs/train_continual.json` will overwrite ones in `configs/train.json`. Configs in the back will overide the previous ones if they have the same key.

Single-GPU:

```bash
python -m monai.bundle run \
    --config_file="['configs/train.json','configs/train_continual.json']"
```

Multi-GPU:

```bash
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
python -m monai.bundle run \
    --config_file="['configs/train.json','configs/train_continual.json','configs/evaluate.json']"
```

Multi-GPU:

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m monai.bundle run \
    --config_file="['configs/train.json','configs/train_continual.json','configs/evaluate.json','configs/mgpu_evaluate.json']"
```

### Evaluation explanatory items

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

## Configurations

### Best practice to set label_mapping

For a class that represent the same or similar class as the global index, directly map it to the global index. For example, "mouse left lung" (e.g. index 2 in the mouse dataset) can be mapped to the 28 "left lung upper lobe"(or 29 "left lung lower lobe") with [[2,28]]. After finetuning, 28 now represents "mouse left lung" and will be used for segmentation. If you want to segment 4 substructures of aorta, you can map one of the substructuress to 6 aorta and the rest to any value, [[1,6],[2,133],[3,134],[4,135]].

```text
NOTE: Do not map to global index value >= 255. `num_classes=255` in the config only represent the maximum mapping index, while the actual output class number only depends on your label_mapping definition. The 255 value in the inference output is also used to represent 'NaN' value.
```

### `val_at_start`

Default `true`, VISTA3D will perform out-of-the-box segmentation before the training.
Users can disable if the validation takes too long.

### `n_train_samples` and `n_val_samples`

In `train_continual.json`, only `n_train_samples` and `n_val_samples` are used for training and validation.

### `patch_size`

The patch size parameter is defined in `configs/train_continual.json`: `"patch_size": [128, 128, 128]`. For finetuning purposes, this value needs to be changed acccording to user's task and GPU memory. Usually a larger patch_size will give better final results. `[192,192,128]` is a good value for larger memory GPU.

### `resample_to_spacing`

The resample_to_spacing parameter is defined in `configs/train_continual.json` and it represents the resolution the model will be trained on. The `1.5,1.5,1.5` mm default is suitable for large CT organs, but for other tasks, this value should be changed to achive the optimal performance.

### Advanced user: `drop_label_prob` and `drop_point_prob` (in train.json)

VISTA3D is trained to perform both automatic (class prompts) and interactive point segmentation.
`drop_label_prob` and `drop_point_prob` means percentage to remove class prompts and point prompts during training respectively. If `drop_point_prob=1`, the
model is only finetuning for automatic segmentation, while `drop_label_prob=1` means only finetuning for interactive segmentation. The VISTA3D foundation
model is trained with interactive only (drop_label_prob=1) and then froze the point branch and trained with fully automatic segmentation (`drop_point_prob=1`).
In this bundle, the training is simplified by jointly training with class prompts and point prompts and both of the drop ratio is set to 0.25.

```text
NOTE: If user doesn't use interactive segmentation, set `drop_point_prob=1` and `drop_label_prob=0` in train.json might provide a faster and easier finetuning process.
```

### Training explanatory items

In `train.json`, `validate[evaluator][val_head]` can be `auto` and `point`. If `auto`, the validation results will be automatic segmentation. If `point`,
the validation results will be sampling one positive point per object per patch. The validation scheme of combining auto and point is deprecated due to
speed issue.

In `train_continual.json`, `valid_remap` is a transform that maps the groundtruth label indexes, e.g. [0,2,3,5,6] to sequential and continuous labels [0,1,2,3,4]. This is
required by monai dice calculation. It is not related to mapping label index to VISTA3D defined global class index. The validation data is not mapped
to the VISTA3D global class index.

`label_set` is used to identify the VISTA model classes for providing training prompts.
`val_label_set` is used to identify the original training label classes for computing foreground/background mask during validation.
The default configs for both variables are derived from the `label_mappings` config and include `[0]`:

```json
"label_set": "$[0] + list(x[1] for x in @label_mappings#default)"
"val_label_set": "$[0] + list(x[0] for x in @label_mappings#default)"
```

Note: Please ensure the input data header is correct. The output file will use the same header as the input data, but if the input data is missing header information, MONAI will automatically provide some default values for missing values (e.g. `np.eye(4)` will be used if affine information is absent). This may cause a visualization misalignment depending on the visualization tool.
