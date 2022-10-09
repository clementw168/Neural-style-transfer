from src.dataset.loading import load_image_from_dataset, resize_pil_image
from src.dataset.models import ImageCategory, StyleCategory
from src.image_based import run_image_based_training_loop
from src.image_based.models import SeedType
from src.loss.models import NetworkAchitecture
from src.utils.models import DeviceType
from src.utils.optimizers import OptimizerType

if __name__ == "__main__":
    content_image = load_image_from_dataset(ImageCategory.CONTENT, index=1)
    content_image = resize_pil_image(content_image, 512)
    style_image = load_image_from_dataset(
        ImageCategory.STYLE, style=StyleCategory.IMPRESSIONISM, index=4
    )
    # style_image = content_image

    run_image_based_training_loop(
        raw_content_image=content_image,
        raw_style_image=style_image,
        seed_type=SeedType.CONTENT_IMAGE,
        steps=3000,
        logger_update_steps=10,
        content_coefficient=8,  # 8
        style_coefficient=100,
        total_variation_coefficient=0,  # 1
        learning_rate=0.1,  # 0.1
        optimizer_type=OptimizerType.ADAM,
        network_architecture=NetworkAchitecture.VGG19,
        device=DeviceType.GPU,
        zoom_loss=True,
    )
