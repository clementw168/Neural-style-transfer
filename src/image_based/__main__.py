from src.dataset.loading import load_image_from_dataset
from src.dataset.models import ImageCategory, StyleCategory
from src.image_based import run_image_based_training_loop
from src.image_based.models import SeedType
from src.utils.models import DeviceType
from src.utils.optimizers import OptimizerType

if __name__ == "__main__":
    content_image = load_image_from_dataset(ImageCategory.CONTENT, index=0)
    style_image = load_image_from_dataset(
        ImageCategory.STYLE, style=StyleCategory.IMPRESSIONISM, index=0
    )

    run_image_based_training_loop(
        raw_content_image=content_image,
        raw_style_image=style_image,
        seed_type=SeedType.RANDOM,
        steps=1000,
        logger_update_steps=10,
        content_coefficient=1e2,
        style_coefficient=0,
        total_variation_coefficient=None,
        learning_rate=1,
        optimizer_type=OptimizerType.ADAM,
        device=DeviceType.GPU,
    )
