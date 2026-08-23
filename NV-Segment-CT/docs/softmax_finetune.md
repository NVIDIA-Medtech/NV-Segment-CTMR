# Softmax Finetuning

This workflow converts the pretrained VISTA3D automatic branch into a conventional
fixed-channel segmentation network. It uses mutually exclusive softmax outputs and
does not include interactive point prompts.

## Label mapping

Set `label_mappings` in both `configs/train_continual_softmax.json` and
`configs/inference_softmax.json`. Each entry is:

```text
[dataset label, pretrained VISTA class ID]
```

For example:

```json
"label_mappings": {
    "default": [
        [1, 3],
        [2, 13]
    ]
}
```

This creates three output channels:

```text
channel 0: trainable background classifier
channel 1: dataset label 1, initialized from VISTA class 3
channel 2: dataset label 2, initialized from VISTA class 13
```

Training remaps dataset labels to contiguous channel indices. Inference maps the
channel indices back to the original dataset labels before saving the segmentation.
The mapping order must remain unchanged when using the resulting checkpoint.

## Training

Place the pretrained VISTA3D checkpoint at:

```text
models/model.pt
```

The training data list uses the MONAI Decathlon format. Update
`data_list_file_path`, `dataset_dir`, and `label_mappings` in
`configs/train_continual_softmax.json`, or override the paths on the command line.

Run the self-contained training configuration from the bundle directory:

```bash
python -m monai.bundle run \
  --config_file=configs/train_continual_softmax.json \
  --dataset_dir=/path/to/dataset
```

The default recipe uses 128×128×128 patches, 100 epochs, five warmup epochs,
cached preprocessing, Dice plus cross-entropy loss, and full-network finetuning.
The best validation checkpoint is saved as:

```text
models/model_softmax.pt
```

### Multi-GPU training

Compose the same self-contained softmax configuration with the existing
distributed-training overlay. For example, to train on two GPUs on one node:

```bash
torchrun --standalone --nproc_per_node=2 -m monai.bundle run \
  --config_file="['configs/train_continual_softmax.json','configs/multi_gpu_train.json']" \
  --dataset_dir=/path/to/dataset
```

Each GPU processes the configured per-GPU batch size. Rank 0 writes the checkpoint
and experiment logs.

## Inference

Keep the same `label_mappings` and ordering in `configs/inference_softmax.json`.
Run inference on one image:

```bash
python -m monai.bundle run \
  --config_file=configs/inference_softmax.json \
  --input_file=/path/to/image.nii.gz \
  --checkpoint_path=models/model_softmax.pt \
  --output_dir=eval
```

The segmentation is restored to the input image space and saved under:

```text
eval/<image_name>/<image_name>_seg.nii.gz
```

The saved voxel values use the original dataset labels from the first column of
`label_mappings`. This inference configuration loads only a softmax-finetuned
checkpoint; it does not download a pretrained checkpoint from Hugging Face.

## Important constraints

- The output channels are fixed by `label_mappings` when the model is constructed.
- Training and inference must use the same mapping entries in the same order.
- This workflow is intended for mutually exclusive semantic-segmentation labels.
- Use the standard VISTA3D configs instead if interactive point prompts or dynamic
  label prompts are required.
