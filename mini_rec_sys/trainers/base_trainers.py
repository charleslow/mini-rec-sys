from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Sampler
import pytorch_lightning as pl

from mini_rec_sys.data import Session, SessionDataset, BatchedSequentialSampler
from mini_rec_sys.evaluators import Evaluator


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
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.sampler = sampler
        self.model_params = model_params
        self.optimizer_class = optimizer_class
        self.learning_rate = learning_rate
        self.val_dataset = val_dataset
        self.val_batch_size = val_batch_size

    def forward(self):
        """
        Each child class should implement this method, which should load one
        mini batch of training data and generate a single loss value, which will be
        used by the Trainer to compute gradients against.
        """
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch: list[dict], batch_idx: int):
        loss = self.forward(batch)
        self.log("loss", loss.item(), prog_bar=True)
        return loss

    def validation_step(self, batch: list[dict], batch_idx):
        """
        Child classes can override this method if a custom validation is required,
        i.e. we want the validation to be different from training.
        """
        loss = self.forward(batch)
        self.log("val_loss", loss.item(), prog_bar=True, batch_size=len(batch))
        return loss


def train(model: BaseModel, **kwargs):
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

    trainer = pl.Trainer(**kwargs)
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
