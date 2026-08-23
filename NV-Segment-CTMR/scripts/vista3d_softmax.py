"""Automatic-only VISTA3D for multiclass softmax segmentation."""

from collections.abc import Sequence

import torch
from monai.networks.nets import SegResNetDS2
from monai.networks.nets.vista3d import ClassMappingClassify
from torch import nn


class Vista3dSoftmax(nn.Module):
    """Return one learned background channel plus the selected VISTA classes."""

    def __init__(
        self,
        in_channels: int = 1,
        class_ids: Sequence[int] = (1,),
        feature_size: int = 48,
    ) -> None:
        super().__init__()
        if not class_ids:
            raise ValueError("class_ids must contain at least one foreground class")

        self.image_encoder = SegResNetDS2(
            in_channels=in_channels,
            blocks_down=(1, 2, 2, 4, 4),
            norm="instance",
            out_channels=feature_size,
            init_filters=feature_size,
            dsdepth=1,
        )

        # Keep the decoder depth metadata, but discard the unused point decoder.
        self.image_encoder.up_layers = nn.ModuleList(nn.Identity() for _ in self.image_encoder.up_layers)
        self.class_head = ClassMappingClassify(
            n_classes=512,
            feature_size=feature_size,
            use_mlp=True,
        )
        self.background_classifier = nn.Parameter(torch.empty(1, feature_size))
        nn.init.normal_(self.background_classifier, std=0.02)
        self.register_buffer(
            "class_ids",
            torch.as_tensor(class_ids, dtype=torch.long).reshape(-1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, features = self.image_encoder(images, with_point=False, with_label=True)
        features = self.class_head.image_post_mapping(features)

        foreground = self.class_head.class_embeddings(self.class_ids)
        foreground = self.class_head.mlp(foreground).squeeze(1)
        classifiers = torch.cat([self.background_classifier, foreground])

        return torch.einsum("kc,bchwd->bkhwd", classifiers, features)
