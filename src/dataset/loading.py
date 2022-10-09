import os

from PIL import Image
from pydantic import NonNegativeInt, PositiveInt

from src.dataset.models import ImageCategory, StyleCategory
from src.utils.constants import DATASET_ROOT


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_image_from_dataset(
    image_category: ImageCategory,
    index: NonNegativeInt | None,
    style: StyleCategory | None = None,
) -> Image.Image:
    path = os.path.join(
        DATASET_ROOT,
        image_category,
    )
    if style is not None:
        path = os.path.join(path, style)
    files = os.listdir(path)
    if index is None or index >= len(files):
        index = 0
    path = os.path.join(path, files[index])

    return load_image(path)


def resize_pil_image(image: Image.Image, max_size: PositiveInt) -> Image.Image:
    width, height = image.size
    if width > height:
        height = int(max_size * height / width)
        width = max_size
    else:
        width = int(max_size * width / height)
        height = max_size
    return image.resize((width, height))


if __name__ == "__main__":
    image = load_image_from_dataset(
        ImageCategory.STYLE, style=StyleCategory.IMPRESSIONISM, index=0
    )
    image.show()
