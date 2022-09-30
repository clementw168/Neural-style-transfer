from typing import Callable

import torch

from src.loss.models import NetworkAchitecture
from src.loss.networks.vgg import VGGExtractor
from src.utils.models import DeviceType


class PerceptualLoss:
    def __init__(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        content_coefficient: float,
        style_coefficient: float,
        total_variation_coefficient: float | None = None,
        network_architecture: NetworkAchitecture = NetworkAchitecture.VGG19,
        device: DeviceType = DeviceType.GPU,
        requires_grad: bool = False,
    ):
        self.content_coefficient = content_coefficient
        self.style_coefficient = style_coefficient
        self.total_variation_coefficient = total_variation_coefficient
        self.device = device
        self.loss_layers = network_architecture.get_perceptual_layers()
        self.features_extractor = self._load_extractor(
            network_architecture=network_architecture,
            layers=self.loss_layers.CONTENT | self.loss_layers.STYLE,
            requires_grad=requires_grad,
        ).to(device)

        self.content_representation = self._get_content_representation(
            content_image=content_image.to(device)
        )

        self.style_representation = self._get_style_representation(
            style_image=style_image.to(device)
        )

    @staticmethod
    def _load_extractor(
        network_architecture: NetworkAchitecture,
        layers: set[str],
        requires_grad: bool = False,
    ) -> torch.nn.Module:
        match network_architecture:
            case NetworkAchitecture.VGG19:
                return VGGExtractor(
                    network_architecture, layers, requires_grad=requires_grad
                )

            case NetworkAchitecture.VGG16:
                return VGGExtractor(
                    network_architecture, layers, requires_grad=requires_grad
                )

            case _:
                raise NotImplementedError(
                    f"Network {network_architecture} not supported yet"
                )

    def _get_style_representation(self, style_image: torch.Tensor):
        features = self.features_extractor(style_image)

        return {
            layer: self.gram_matrix(features[layer]) for layer in self.loss_layers.STYLE
        }

    def _get_content_representation(self, content_image: torch.Tensor):
        features = self.features_extractor(content_image)

        return {layer: features[layer] for layer in self.loss_layers.CONTENT}

    def content_loss(
        self, content_features: torch.Tensor, generated_features: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(torch.square(content_features - generated_features))

    def style_loss(
        self, style_gram_matrix: torch.Tensor, generated_features: torch.Tensor
    ):
        return torch.mean(
            torch.square(style_gram_matrix - self.gram_matrix(generated_features))
        )

    def total_variation_loss(self, generated_image: torch.Tensor) -> torch.Tensor:
        return (
            torch.mean(
                torch.square(
                    generated_image[:, :, :, :-1] - generated_image[:, :, :, 1:]
                )
                + torch.square(
                    generated_image[:, :, :-1, :] - generated_image[:, :, 1:, :]
                )
            )
            * self.total_variation_coefficient
        )

    @staticmethod
    def gram_matrix(style_features: torch.Tensor) -> torch.Tensor:
        channels, height, width = style_features.size()
        features = style_features.view(channels, height * width)
        features_t = features.transpose(1, 0)
        gram = features.mm(features_t)

        return gram.div(channels * height * width)

    def forward(
        self, generated_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        generated_features = self.features_extractor(generated_image)
        content_loss = torch.zeros(1).to(self.device)
        style_loss = torch.zeros(1).to(self.device)

        for layer in self.loss_layers.CONTENT:
            content_loss += self.content_loss(
                self.content_representation[layer], generated_features[layer]
            )
        content_loss *= self.content_coefficient / len(self.loss_layers.CONTENT)

        for layer in self.loss_layers.STYLE:
            style_loss += self.style_loss(
                self.style_representation[layer], generated_features[layer]
            )
        style_loss *= self.style_coefficient / len(self.loss_layers.STYLE)

        total_loss = content_loss + style_loss

        if self.total_variation_coefficient:
            total_variation_loss = self.total_variation_loss(generated_image)
            total_loss += total_variation_loss
        else:
            total_variation_loss = None

        return total_loss, content_loss, style_loss, total_variation_loss

    def __call__(
        self, generated_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.forward(generated_image)
