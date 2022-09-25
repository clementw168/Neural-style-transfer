from torch.cuda import is_available

from src.utils.models import DeviceType

DATASET_ROOT: str = "./dataset"

device: DeviceType = DeviceType.GPU if is_available() else DeviceType.CPU
