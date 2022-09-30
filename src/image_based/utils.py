import torch

from src.dataset.constants import NORMALIZING_MEAN, NORMALIZING_STD
from src.image_based.models import SeedType
from src.utils.models import DeviceType


def get_image_seed(
    seed_type: SeedType,
    content_image: torch.Tensor,
    style_image: torch.Tensor,
    device: DeviceType = DeviceType.GPU,
) -> torch.Tensor:
    match seed_type:
        case SeedType.CONTENT_IMAGE:
            seed_image = content_image

        case SeedType.STYLE_IMAGE:
            seed_image = style_image

        case SeedType.RANDOM:
            seed_image = (
                torch.randn_like(content_image)
                - torch.tensor(NORMALIZING_MEAN).view(3, 1, 1)
            ).div(torch.tensor(NORMALIZING_STD).view(3, 1, 1))

        case _:
            raise NotImplementedError(f"Seed type {seed_type} not supported yet")

    return seed_image.clone().detach().to(device).requires_grad_(True)
