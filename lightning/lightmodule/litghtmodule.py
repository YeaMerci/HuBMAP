import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, JaccardIndex, FBetaScore
from typing import Any, Union
import wandb


class HuBMAPLightningModule(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 model: nn.Module,
                 lr: float = None,
                 ):

        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.zeros(size=[1, 3, 512, 512])
        self.num_classes = num_classes
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self._device = "cuda"

        self.step_outputs = {
            "loss": [],
            "accuracy": [],
            "jaccard_index": [],
            "fbeta_score": []
        }

        self.metrics = {
            "accuracy": Accuracy(task="multiclass",
                                 threshold=0.5,
                                 num_classes=num_classes,
                                 validate_args=True,
                                 ignore_index=None,
                                 average="micro").to(self._device),

            "jaccard_index": JaccardIndex(task="multiclass",
                                          threshold=0.5,
                                          num_classes=num_classes,
                                          validate_args=True,
                                          ignore_index=None,
                                          average="macro").to(self._device),

            "fbeta_score": FBetaScore(task="multiclass",
                                      beta=1.0,
                                      threshold=0.5,
                                      num_classes=num_classes,
                                      average="micro",
                                      ignore_index=None,
                                      validate_args=True).to(self._device)
        }

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch

        assert x.ndim == 4
        assert x.max() <= 3 and x.min() >= -3
        assert y.ndim == 3
        assert y.max() <= 22 and y.min() >= 0

        logites = self.forward(x.to(torch.float32))
        activated = F.softmax(input=logites, dim=1)
        predictions = torch.argmax(activated, dim=1)
        loss = self.criterion(logites, y.to(torch.int64))

        accuracy = self.metrics["accuracy"](predictions, y)
        jaccard_index = self.metrics["jaccard_index"](predictions, y)
        fbeta_score = self.metrics["fbeta_score"](predictions, y)

        self.step_outputs["loss"].append(loss)
        self.step_outputs["accuracy"].append(accuracy)
        self.step_outputs["jaccard_index"].append(jaccard_index)
        self.step_outputs["fbeta_score"].append(fbeta_score)
        return loss

    def shared_epoch_end(self, stage: Any):
        loss = torch.mean(torch.tensor([
            loss for loss in self.step_outputs["loss"]
        ]))

        accuracy = torch.mean(torch.tensor([
            accuracy for accuracy in self.step_outputs["accuracy"]
        ]))

        jaccard_index = torch.mean(torch.tensor([
            jaccard_index for jaccard_index in self.step_outputs["jaccard_index"]
        ]))

        fbeta_score = torch.mean(torch.tensor(
            [fbeta_score for fbeta_score in self.step_outputs["fbeta_score"]
             ]))

        for key in self.step_outputs.keys():
            self.step_outputs[key].clear()

        metrics = {
            f"{stage}_loss": loss,
            f"{stage}_accuracy": accuracy,
            f"{stage}_jaccard_index": jaccard_index,
            f"{stage}_fbeta_score": fbeta_score
        }

        wandb.log(metrics)
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="train")

    def on_train_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="train")

    def validation_step(self, batch: Any, batch_idx: Any):
        return self.shared_step(batch=batch, stage="val")

    def on_validation_epoch_end(self) -> None:
        return self.shared_epoch_end(stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr
        )

        scheduler_dict = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=5
            ),
            "interval": "epoch",
            "monitor": "val_loss"
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
