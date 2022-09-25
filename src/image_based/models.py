from enum import Enum


class SeedType(str, Enum):
    CONTENT_IMAGE = "content_image"
    RANDOM = "random"
    STYLE_IMAGE = "style_image"
