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

## Resources

- **Quick Deployment**: [NVIDIA NIM for NV-Segment-CT](https://build.nvidia.com/nvidia/vista-3d) - Managed API endpoint
- **Documentation**: See individual model folders for detailed docs
- **Research Paper**: [VISTA3D: Versatile Imaging SegmenTation and Annotation](https://arxiv.org/abs/2406.05285)
- **Built with**: [MONAI](https://monai.io/) - Medical Open Network for AI
