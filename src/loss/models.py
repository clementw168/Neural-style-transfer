from enum import Enum
from typing import Type


class PerceptualLayers(set[str], Enum):
    CONTENT: set[str]
    STYLE: set[str]


class NetworkAchitecture(str, Enum):
    VGG19 = "vgg19"
    VGG16 = "vgg16"
    RESNET50 = "resnet50"
    INCEPTIONV3 = "inceptionv3"
    XCEPTION = "xception"

    def get_perceptual_layers(self) -> Type[PerceptualLayers]:
        match self:
            case NetworkAchitecture.VGG19:
                return VGG19LossLayers

            case NetworkAchitecture.VGG16:
                return VGG16LossLayers

            case _:
                raise NotImplementedError(f"Network {self} not supported yet")


class VGG19LossLayers(PerceptualLayers):
    STYLE = {"conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"}
    CONTENT = {"conv4_2"}


class VGG16LossLayers(PerceptualLayers):
    STYLE = {"conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"}
    CONTENT = {"conv5_2"}
