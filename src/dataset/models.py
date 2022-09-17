from enum import Enum


class ImageCategory(str, Enum):
    CONTENT = "content"
    STYLE = "style"


class StyleCategory(str, Enum):
    ANCIENT_ART = "ancient_art"
    IMPRESSIONISM = "impressionism"
    MEDIEVAL = "medieval"
    PREHISTORIC = "prehistoric"
    RENAISSANCE = "renaissance"
