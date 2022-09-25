from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "cuda"
