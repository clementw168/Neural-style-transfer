from typing import Callable, Iterable

import torch
from PIL.Image import Image
from pydantic import PositiveInt
from torchvision import transforms

from src.dataset.constants import NORMALIZING_MEAN, NORMALIZING_STD


class Unnormalize:
    def __init__(self, mean: Iterable, std: Iterable):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean


def get_preprocessing_transforms(
    shape: tuple[PositiveInt, PositiveInt]
) -> Callable[[Image], torch.Tensor]:
    return transforms.Compose(
        [
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZING_MEAN, std=NORMALIZING_STD),
        ]
    )  # type: ignore


def get_postprocessing_transforms() -> Callable[[torch.Tensor], Image]:
    return transforms.Compose(
        [
            Unnormalize(mean=NORMALIZING_MEAN, std=NORMALIZING_STD),
            transforms.ToPILImage(),
        ]
    )  # type: ignore
