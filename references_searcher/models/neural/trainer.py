from tqdm import tqdm
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import wandb

from typing import Literal

from references_searcher.utils import generate_device, log_with_message
from references_searcher import logger
from references_searcher.metrics import MetricCalculator


class Trainer:
    def __init__(
        self,
        watcher: Literal["wandb"] | None,
        criterion: torch.nn.Module = nn.CrossEntropyLoss(),
        device: torch.device | None = None,
    ) -> None:

        self.metric_calculator = MetricCalculator()
        self.criterion = criterion
        self.device = device
        self.watcher = watcher
        self.device = device if device is not None else generate_device()

        if watcher == "wandb":
            self._watcher_command = wandb.log
        elif watcher is not None:
            message = "Watchers except WandB are not implemented yet."
            logger.error(message)
            raise NotImplementedError(message)
        else:
            self._watcher_command = lambda *_: None

        self.optimizer = None
        self.model = None

    def train(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        n_epochs: int = 10,
        val_dataloader: torch.utils.data.DataLoader | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        verbose: bool = True,
        watch: bool = True,
    ) -> None:

        for epoch in range(1, n_epochs + 1):
            logger.info(f"Starting epoch {epoch}/{n_epochs}.")
            # Train
            epoch_train_state = self._train_single_epoch(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_dataloader,
                verbose=verbose,
            )

            current_epoch_scores = {"train": epoch_train_state}

            if scheduler is not None:
                scheduler.step()

            # Validation
            if val_dataloader is not None:
                epoch_val_scores = self.evaluate(
                    model=model,
                    val_dataloader=val_dataloader,
                    verbose=verbose,
                )
                current_epoch_scores["val"] = epoch_val_scores

            current_epoch_scores["train_epoch"] = epoch
            if watch:
                try:
                    self._watcher_command(current_epoch_scores)
                except Exception as e:
                    logger.error(f"Error loading to watcher after train at epoch {epoch}!")
                    raise e

            logger.info(f"Finished epoch {epoch}/{n_epochs}.")

    def _train_single_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: torch.utils.data.DataLoader,
        verbose: bool,
    ) -> dict[str, float]:

        logger.info("Started training the model.")
        model.train()
        pbar = tqdm(train_dataloader, leave=False, desc="Training model") if verbose else train_dataloader
        self.metric_calculator.reset()
        train_loss = 0.0
        scaler = GradScaler()

        for items_batch, targets_batch in pbar:
            items_batch = {key: tensor.to(self.device) for key, tensor in items_batch.items()}
            targets_batch = targets_batch.to(self.device)

            optimizer.zero_grad()
            with autocast():
                predicted_logits = model(**items_batch, return_probabilities=False)
                loss = self.criterion(predicted_logits, targets_batch)

            # We use running loss and scores to avoid double-running through the train dataset. This does not
            #   provide objective scores, but is good enough compared to time-effectiveness.
            train_loss += loss.item() * len(targets_batch)
            predicted_classes = torch.argmax(predicted_logits, dim=1)
            self.metric_calculator.update(predicted_classes, targets_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        result = {
            "train_loss": train_loss / len(train_dataloader.dataset),
            **self.metric_calculator.calculate_metrics(),
        }

        logger.info("Finished training the model.")
        logger.debug(f"Model train scores: {result}")

        return result

    def pretrain(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        pretrain_dataloader: torch.utils.data.DataLoader,
        pretrain_criterion: torch.nn.Module = torch.nn.TripletMarginLoss(),
        n_epochs: int = 10,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        verbose: bool = True,
        watch: bool = True,
    ) -> None:

        for epoch in range(1, n_epochs + 1):
            with log_with_message(f"epoch {epoch}/{n_epochs}"):
                # Train
                epoch_pretrain_state = self._pretrain_single_epoch(
                    model=model,
                    pretrain_optimizer=optimizer,
                    pretrain_dataloader=pretrain_dataloader,
                    pretrain_criterion=pretrain_criterion,
                    verbose=verbose,
                )

                current_epoch_scores = {"pretrain": epoch_pretrain_state, "pretrain_epoch": epoch}

                if scheduler is not None:
                    scheduler.step()

                if watch:
                    try:
                        self._watcher_command(current_epoch_scores)
                    except Exception as e:
                        logger.error(f"Error loading to watcher after train at epoch {epoch}!")
                        raise e

    def _pretrain_single_epoch(
        self,
        model: torch.nn.Module,
        pretrain_optimizer: torch.optim.Optimizer,
        pretrain_dataloader: torch.utils.data.DataLoader,
        pretrain_criterion: torch.nn.Module,
        verbose: bool,
    ) -> dict[str, float]:

        model.train()
        pbar = tqdm(pretrain_dataloader, leave=False, desc="Pretraining model") if verbose else pretrain_dataloader
        pretrain_loss = 0.0
        scaler = GradScaler()

        for items_batch in pbar:
            items_batch = {key: tensor.to(self.device) for key, tensor in items_batch.items()}

            pretrain_optimizer.zero_grad()
            with autocast():
                anchor_embedding, positive_embedding, negative_embedding = model.triplet(**items_batch)
                loss = pretrain_criterion(anchor_embedding, positive_embedding, negative_embedding)

            # We use running loss to avoid double-running through the train dataset. This does not
            #   provide objective scores, but is good enough compared to time-effectiveness.
            pretrain_loss += loss.item() * len(anchor_embedding)

            scaler.scale(loss).backward()
            scaler.step(pretrain_optimizer)
            scaler.update()

        pretrain_loss /= len(pretrain_dataloader.dataset)
        logger.debug(f"Model pretrain loss: {pretrain_loss}")

        return {
            "pretrain_loss": pretrain_loss,
        }

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        val_dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> dict[str, float]:

        logger.info("Started validating the model.")
        model.eval()
        pbar = tqdm(val_dataloader, leave=False, desc="Validating model") if verbose else val_dataloader
        self.metric_calculator.reset()
        val_loss = 0.0

        for items_batch, targets_batch in pbar:
            items_batch = {key: tensor.to(self.device) for key, tensor in items_batch.items()}
            targets_batch = targets_batch.to(self.device)

            predicted_logits = model(**items_batch, return_probabilities=False)
            loss = self.criterion(predicted_logits, targets_batch)

            val_loss += loss.item() * len(targets_batch)
            predicted_classes = torch.argmax(predicted_logits, dim=1)
            self.metric_calculator.update(predicted_classes, targets_batch)

        result = {
            "val_loss": val_loss / len(val_dataloader.dataset),
            **self.metric_calculator.calculate_metrics(),
        }

        logger.info("Finished validating the model.")
        logger.debug(f"Model val scores: {result}")
        return result
