from torch.cuda import is_available

DATASET_ROOT = "./dataset"

device = "cuda" if is_available() else "cpu"
