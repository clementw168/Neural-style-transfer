import os
from datetime import datetime

import matplotlib.pyplot as plt
from PIL.Image import Image
from pydantic import PositiveInt


class TrainingLogger:
    def __init__(
        self,
        update_steps: PositiveInt = 1,
        log_path: str | None = None,
        zoom_loss: bool = True,
    ):
        self.update_steps = update_steps
        self.zoom_loss = zoom_loss

        self.log_path = (
            log_path
            if log_path is not None
            else f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.reset()

    def reset(self):
        self.content_loss = []
        self.style_loss = []
        self.total_variation_loss = []
        self.total_loss = []
        self.steps = []

    def update(
        self,
        image: Image,
        step: PositiveInt,
        total_loss: float,
        content_loss: float,
        style_loss: float,
        total_variation_loss: float | None = None,
    ):
        self.steps.append(step)
        self.total_loss.append(total_loss)
        self.content_loss.append(content_loss)
        self.style_loss.append(style_loss)
        if total_variation_loss is not None:
            self.total_variation_loss.append(total_variation_loss)

        if (step + 1) % self.update_steps == 0:
            self.log(
                image, step, total_loss, content_loss, style_loss, total_variation_loss
            )

    def log(
        self,
        image: Image,
        step: PositiveInt,
        total_loss: float,
        content_loss: float,
        style_loss: float,
        total_variation_loss: float | None = None,
    ):
        total_variation_loss = (
            0.0 if total_variation_loss is None else total_variation_loss
        )
        print(
            f"Step {step}: Total loss: {total_loss:.5f}, Content loss: {content_loss:.5f}, Style loss: {style_loss:.5f}, Total variation loss: {total_variation_loss:.5f}"
        )
        self.plot_loss(zoom=self.zoom_loss)

        self.save_image(image, step)

    def plot_loss(self, zoom: bool = False):
        plt.plot(self.steps, self.total_loss, label="Total loss")
        plt.plot(self.steps, self.content_loss, label="Content loss")
        plt.plot(self.steps, self.style_loss, label="Style loss")
        if len(self.total_variation_loss) > 0:
            plt.plot(
                self.steps, self.total_variation_loss, label="Total variation loss"
            )
        plt.legend()
        plt.savefig(f"{self.log_path}/loss.png")
        plt.close()

        if zoom and len(self.total_loss) > 500:
            plt.plot(self.steps[500:], self.total_loss[500:], label="Total loss")
            plt.plot(self.steps[500:], self.content_loss[500:], label="Content loss")
            plt.plot(self.steps[500:], self.style_loss[500:], label="Style loss")
            if len(self.total_variation_loss) > 0:
                plt.plot(
                    self.steps[500:],
                    self.total_variation_loss[500:],
                    label="Total variation loss",
                )
            plt.legend()
            plt.savefig(f"{self.log_path}/loss_zoom.png")
            plt.close()

    def save_image(self, image: Image, step: PositiveInt | str):
        image.save(f"{self.log_path}/image_{step}.png")
