import torch

from src.loss.image_based_loss import PerceptualLoss


def training_step(
    generated_image: torch.Tensor,
    loss_function: PerceptualLoss,
    optimizer: torch.optim.Optimizer,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    total_loss, content_loss, style_loss, total_variation_loss = loss_function(
        generated_image
    )

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return (
        total_loss.cpu(),
        content_loss.cpu(),
        style_loss.cpu(),
        total_variation_loss.cpu() if total_variation_loss else None,
    )
