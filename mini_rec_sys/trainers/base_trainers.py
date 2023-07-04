from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Sampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

from mini_rec_sys.data import Session, SessionDataset, BatchedSequentialSampler
from mini_rec_sys.evaluators import mean_with_se
from mini_rec_sys.constants import (
    VAL_METRIC_NAME,
    TRAIN_METRIC_NAME,
    TEST_METRIC_NAME,
    TEST_SE_NAME,
    TEST_N_NAME,
)
from pdb import set_trace


class BaseModel(pl.LightningModule):
    """
    Trainers train encoders based on some type of training collection and loss.
    It also optionally:
    - logs some validation / test performance
    - saves the best model

    # TODO: Add gradient accumulation option.
    """

    def __init__(
        self,
        train_dataset: SessionDataset,
        sampler: Sampler,
        model_params: dict = None,
        optimizer_class: Optimizer = Adam,
        learning_rate: float = 1e-5,
        val_dataset: SessionDataset = None,
        val_batch_size: int = 32,
        test_dataset: SessionDataset = None,
        test_batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.sampler = sampler
        self.model_params = model_params
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.val_dataset = val_dataset
        self.val_batch_size = val_batch_size
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size
        self.test_metrics = []

    def forward(self):
        """
        Each child class should implement this method, which should load one
        mini batch of training data and generate a loss scalar.
        It will be used by the Trainer to compute gradients.
        """
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: list[dict], batch_idx: int):
        loss = self.forward(batch)
        self.log(TRAIN_METRIC_NAME, loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch: list[dict], batch_idx):
        """
        Child classes can override this method if a custom validation is required,
        i.e. we want the validation to be different from training.
        """
        loss = self.forward(batch)
        self.log(VAL_METRIC_NAME, loss.item(), prog_bar=True, batch_size=len(batch))
        return loss

    def test_step(self, batch: list[dict], batch_idx):
        """
        Child classes can override this method if a custom test step is required,
        i.e. we want the test to be different from training.

        Note that the test step should accumulate batch metrics to
        self.test_metrics, which will be used to compute the mean and standard
        error at the end of the epoch.
        """
        loss = self.forward(batch)
        self.test_metrics.append(loss.item())
        return loss

    def on_test_epoch_end(self):
        metric, se = mean_with_se(self.test_metrics)
        n = len(self.test_metrics)
        self.test_metrics = []
        self.log(TEST_METRIC_NAME, metric, prog_bar=True)
        self.log(TEST_SE_NAME, se, prog_bar=True)
        self.log(TEST_N_NAME, n, prog_bar=True)


def train(
    model: BaseModel,
    max_epochs: int = 20,
    limit_train_batches: int = 100,
    limit_val_batches: int = 20,
    log_every_n_steps: int = 50,
    limit_test_batches: int = 20,
    precision: int | str = 32,
    checkpoint_metric: str = None,
    checkpoint_behaviour: str = "min",
    **kwargs
):
    """
    Additional arguments / keyword arguments are passed into pl.Trainer.
    """
    train_loader = DataLoader(
        model.train_dataset, batch_sampler=model.sampler, collate_fn=lambda x: x
    )

    if model.val_dataset:
        val_loader = DataLoader(
            model.val_dataset,
            batch_sampler=BatchedSequentialSampler(
                model.val_dataset, model.val_batch_size, drop_last=False
            ),
            collate_fn=lambda x: x,
        )
    else:
        val_loader = None

    if model.test_dataset:
        test_loader = DataLoader(
            model.test_dataset,
            batch_sampler=BatchedSequentialSampler(
                model.test_dataset, model.test_batch_size, drop_last=False
            ),
            collate_fn=lambda x: x,
        )
    else:
        test_loader = None

    if checkpoint_metric is None:
        callbacks = []
    else:
        callbacks = [
            ModelCheckpoint(monitor=checkpoint_metric, mode=checkpoint_behaviour)
        ]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps,
        limit_test_batches=limit_test_batches,
        precision=precision,
        callbacks=callbacks,
        **kwargs
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    if test_loader is not None:
        trainer.test(
            dataloaders=test_loader,
            ckpt_path="best",
        )
