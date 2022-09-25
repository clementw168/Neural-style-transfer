import torch
from torch import nn
from torchvision.models import VGG16_Weights, VGG19_Weights, vgg16, vgg19

from src.loss.models import NetworkAchitecture


class VGGExtractor(nn.Module):
    def __init__(
        self,
        network_architecture: NetworkAchitecture,
        layers: set[str],
        requires_grad=False,
    ):
        super().__init__()
        assert network_architecture in {
            NetworkAchitecture.VGG16,
            NetworkAchitecture.VGG19,
        }, "Invalid network model"

        self.submodules, self.output_layers = self._get_submodules(
            network_architecture, layers
        )

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output = {}
        for name, submodule in zip(self.output_layers, self.submodules):
            x = submodule(x)
            output[name] = x

        return output

    @staticmethod
    def _load_vgg_extractor(network_architecture: NetworkAchitecture) -> nn.Module:
        match network_architecture:
            case NetworkAchitecture.VGG19:
                return vgg19(weights=VGG19_Weights.DEFAULT, progress=True).features

            case NetworkAchitecture.VGG16:
                return vgg16(weights=VGG16_Weights.DEFAULT, progress=True).features

            case _:
                raise ValueError(f"Network {network_architecture} not supported")

    @staticmethod
    def _parse_submodules(
        feature_extractor: nn.Module, layers: set[str]
    ) -> tuple[nn.ModuleList, list[str]]:
        output_order = []
        submodules = []
        current_submodule = nn.Sequential()
        block_index = 1
        conv_index = 1

        for index, layer in enumerate(feature_extractor.children()):
            match type(layer):
                case nn.modules.Conv2d:
                    layer_name = f"conv{block_index}_{conv_index}"
                case nn.modules.ReLU:
                    layer_name = f"relu{block_index}_{conv_index}"
                    conv_index += 1
                case nn.modules.MaxPool2d:
                    layer_name = f"pool{block_index}"
                    block_index += 1
                    conv_index = 1
                case _:
                    layer_name = f"layer{index}"
                    ValueError(f"Layer {layer} not supported")

            current_submodule.add_module(str(index), layer)
            if layer_name in layers:
                submodules.append(current_submodule)
                output_order.append(layer_name)
                if len(output_order) == len(layers):
                    break
                current_submodule = nn.Sequential()

        return nn.ModuleList(submodules), output_order

    def _get_submodules(
        self, network_architecture: NetworkAchitecture, layers: set[str]
    ) -> tuple[nn.ModuleList, list[str]]:
        features_extractor = self._load_vgg_extractor(network_architecture)

        return self._parse_submodules(features_extractor, layers)


if __name__ == "__main__":
    from src.loss.models import VGG16LossLayers, VGG19LossLayers

    vgg_extractor = VGGExtractor(NetworkAchitecture.VGG16, VGG16LossLayers.STYLE).to(
        "cuda"
    )
    random_tensor = torch.rand(1, 3, 256, 256).to("cuda")
    output = vgg_extractor(random_tensor)
    print(output)
