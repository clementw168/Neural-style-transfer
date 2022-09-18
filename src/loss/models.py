from enum import Enum


class Network(str, Enum):
    VGG19 = "vgg19"
    VGG16 = "vgg16"
    RESNET50 = "resnet50"
    INCEPTIONV3 = "inceptionv3"
    XCEPTION = "xception"


class VGG19LossLayers(set[str], Enum):
    STYLE = {"conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"}
    CONTENT = {"conv4_2"}


class VGG16LossLayers(set[str], Enum):
    STYLE = {"conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"}
    CONTENT = {"conv5_2"}
