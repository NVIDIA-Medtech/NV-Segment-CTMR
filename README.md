# NV-Segment: Medical Image Segmentation Foundation Models

This repository contains two NVIDIA medical segmentation foundation models for 3D medical image segmentation.

## Model Comparison

Both models follow the MONAI bundle architecture.

| Feature | NV-Segment-CT | NV-Segment-CTMR |
|---------|---------------|-----------------|
| **Anatomical Classes** | 132 classes | 345+ classes |
| **Modalities** | CT only | CT + MRI (body & brain) |
| **Segmentation Type** | Automatic + Interactive (point-click) | Automatic only |
| **Model Weights**     | [NV-Segment-CT on HuggingFace](https://huggingface.co/nvidia/NV-Segment-CT) | [NV-Segment-CTMR on HuggingFace](https://huggingface.co/nvidia/NV-Segment-CTMR) |

**NV-Segment-CT** ([`Paper`](https://arxiv.org/pdf/2406.05285)) is a foundation model trained systematically on 11,454 volumes encompassing 127 types of human anatomical structures and various lesions. The model provides State-of-the-art performances on:

- out-of-the-box automatic segmentation on 3D CT scans
- zero-shot interactive segmentation in 3D CT scans
- automatic segemntation + interactive refinement

**NV-Segment-CTMR** starts from NV-Segment-CT checkpoint and finetuned on over 30K CT and MRI scans, supporting over 300 classes.

- out-of-the-box automatic segmentation on 3D CT scans
- share the same architecture with VISTA3D-CT model but we only trained the automatic segmentation branch with larger CT and MRI datasets.

## Performance on held-out test set

![Benchmark CT](./NV-Segment-CTMR/docs/benchmarkct.png) ![Benchmark MR](./NV-Segment-CTMR/docs/benchmarkmr.png)

## Resources

- **Quick Deployment**: [NVIDIA NIM for NV-Segment-CT](https://build.nvidia.com/nvidia/vista-3d) - Managed API endpoint
- **Documentation**: See individual model folders for detailed docs
- **Research Paper**: [VISTA3D: Versatile Imaging SegmenTation and Annotation](https://arxiv.org/abs/2406.05285)
- **Built with**: [MONAI](https://monai.io/) - Medical Open Network for AI
