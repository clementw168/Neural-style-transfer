import os

from PIL.Image import Image
from pydantic import PositiveFloat, PositiveInt

from src.dataset.processing_images import (
    get_postprocessing_transforms,
    get_preprocessing_transforms,
)
from src.image_based.logger import TrainingLogger
from src.image_based.models import SeedType
from src.image_based.training import training_step
from src.image_based.utils import get_image_seed
from src.loss.image_based_loss import PerceptualLoss
from src.utils.models import DeviceType
from src.utils.optimizers import OptimizerType, get_optimizer


def run_image_based_training_loop(
    raw_content_image: Image,
    raw_style_image: Image,
    seed_type: SeedType,
    steps: PositiveInt,
    content_coefficient: PositiveFloat,
    style_coefficient: PositiveFloat,
    total_variation_coefficient: PositiveFloat | None = None,
    logger_update_steps: PositiveInt = 20,
    learning_rate: PositiveFloat = 0.01,
    optimizer_type: OptimizerType = OptimizerType.ADAM,
    device: DeviceType = DeviceType.GPU,
    save_path: str | None = None,
) -> None:
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    logger = TrainingLogger(log_path=save_path, update_steps=logger_update_steps)

    width, height = raw_content_image.size
    transform = get_preprocessing_transforms((height, width))
    untransform = get_postprocessing_transforms()

    content_image = transform(raw_content_image)
    style_image = transform(raw_style_image)

    generated_image = get_image_seed(seed_type, content_image, style_image, device)

    loss_function = PerceptualLoss(
        content_image=content_image,
        style_image=style_image,
        content_coefficient=content_coefficient,
        style_coefficient=style_coefficient,
        total_variation_coefficient=total_variation_coefficient,
        device=device,
    )

    optimizer = get_optimizer(optimizer_type, [generated_image], learning_rate)

    for step in range(steps):
        total_loss, content_loss, style_loss, total_variation_loss = training_step(
            generated_image, loss_function, optimizer
        )

        logger.update(
            image=untransform(generated_image.clone().detach().cpu()),
            step=step,
            total_loss=total_loss,
            content_loss=content_loss,
            style_loss=style_loss,
            total_variation_loss=total_variation_loss,
        )

    logger.save_image(untransform(generated_image.clone().detach().cpu()), "end")
