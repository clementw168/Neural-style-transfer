import torch

from src.loss.image_based_loss import PerceptualLoss


def training_step(
    generated_image: torch.Tensor,
    loss_function: PerceptualLoss,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float, float, float | None]:
    total_loss, content_loss, style_loss, total_variation_loss = loss_function(
        generated_image
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return (
        total_loss.cpu().item(),
        content_loss.cpu().item(),
        style_loss.cpu().item(),
        total_variation_loss.cpu().item() if total_variation_loss else None,
    )
