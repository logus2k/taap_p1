# mnist_cnn_classifier_calibrated.py
"""
Calibrated CNN classifier for MNIST.
Changes from original:
- Added label_smoothing parameter for calibration
- Added confidence logging to monitor calibration during training
- Logs mean/max confidence and confidence histogram stats
"""

from typing import Optional
import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import MulticlassAccuracy
from lightning.pytorch.utilities.types import OptimizerLRScheduler


class MNISTCNNCalibrated(L.LightningModule):
    """
    Calibrated CNN for MNIST with label smoothing support.
    
    Key calibration parameters:
    - label_smoothing: Softens targets (0.1 recommended for calibration)
    - dropout_p: Adds uncertainty (0.1-0.2 recommended)
    
    Activations: relu | tanh | sigmoid | gelu | leakyrelu | elu
    Optimizers: adam | adamw | sgd | rmsprop
    Scheduler: cosine | none
    """

    def __init__(
        self,
        *,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        activation: str = "relu",
        width: int = 32,
        depth: int = 2,
        dropout_p: float = 0.1,
        use_bn: bool = True,
        weight_init: str = "he",
        num_classes: int = 10,
        # Calibration
        label_smoothing: float = 0.1,
        # Scheduler
        scheduler: str = "none",
        t_max: Optional[int] = None,
        eta_min: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store hyperparameters
        self.lr = float(lr)
        self.optimizer_name = str(optimizer).lower()
        self.weight_decay = float(weight_decay)
        self.act_name = str(activation).lower()
        self.width = int(width)
        self.depth = int(depth)
        assert 1 <= self.depth <= 3, "CNN depth must be in [1, 3]"
        self.dropout_p = float(dropout_p)
        self.use_bn = bool(use_bn)
        self.init_name = str(weight_init).lower()
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)

        self.scheduler_name = str(scheduler).lower()
        self.t_max = None if t_max is None else int(t_max)
        self.eta_min = float(eta_min)

        # Activation factory
        self.act = self._make_activation(self.act_name)

        # Encoder: depth blocks of Conv(3x3) -> BN? -> Act -> MaxPool(2)
        channels = [self.width, 2 * self.width, 2 * self.width][: self.depth]
        in_ch = 1
        blocks: list[nn.Module] = []
        for out_ch in channels:
            blocks.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not self.use_bn))
            if self.use_bn:
                blocks.append(nn.BatchNorm2d(out_ch))
            blocks.append(self._make_activation(self.act_name))
            blocks.append(nn.MaxPool2d(kernel_size=2))
            in_ch = out_ch
        self.encoder = nn.Sequential(*blocks)

        # Head: flatten -> FC(width) -> act -> dropout -> FC(num_classes)
        spatial = 28 // (2 ** self.depth)
        feat_dim = channels[-1] * spatial * spatial
        head_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(feat_dim, self.width),
            self._make_activation(self.act_name),
        ]
        if self.dropout_p > 0.0:
            head_layers.append(nn.Dropout(self.dropout_p))
        head_layers.append(nn.Linear(self.width, self.num_classes))
        self.head = nn.Sequential(*head_layers)

        # Weight initialization
        if self.init_name not in {"he", "xavier"}:
            raise ValueError("weight_init must be 'he' or 'xavier'")
        self.apply(lambda m: self._init_module(m, self.init_name))

        # Loss with label smoothing for calibration
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Metrics
        self.acc_tr = MulticlassAccuracy(num_classes=self.num_classes)
        self.acc_va = MulticlassAccuracy(num_classes=self.num_classes)
        self.acc_te = MulticlassAccuracy(num_classes=self.num_classes)
        
        # Confidence tracking for calibration monitoring
        self._val_confidences = []
        self._val_corrects = []

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        table = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
            "leakyrelu": nn.LeakyReLU,
            "elu": nn.ELU,
        }
        if name not in table:
            raise ValueError(f"Unsupported activation: {name}")
        return table[name]()

    @staticmethod
    def _init_module(m: nn.Module, scheme: str) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if scheme == "he":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "xavier":
                nn.init.xavier_normal_(m.weight)
            else:
                raise ValueError(f"Unknown init scheme: {scheme}")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)
        confidences = probs.max(dim=1).values
        return loss, preds, y, confidences

    def training_step(self, batch, _):
        loss, preds, y, confidences = self._shared_step(batch)
        self.acc_tr.update(preds, y)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_conf_mean", confidences.mean(), on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_end(self):
        acc = self.acc_tr.compute()
        self.log("train_acc", acc, prog_bar=True)
        self.acc_tr.reset()

    def validation_step(self, batch, _):
        loss, preds, y, confidences = self._shared_step(batch)
        self.acc_va.update(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=False)
        
        # Track confidences for calibration analysis
        self._val_confidences.append(confidences.detach())
        self._val_corrects.append((preds == y).detach())
        
        return loss

    def on_validation_epoch_end(self):
        acc = self.acc_va.compute()
        self.log("val_acc", acc, prog_bar=True)
        self.acc_va.reset()
        
        # Compute calibration metrics
        if self._val_confidences:
            all_conf = torch.cat(self._val_confidences)
            all_correct = torch.cat(self._val_corrects)
            
            mean_conf = all_conf.mean().item()
            max_conf = all_conf.max().item()
            min_conf = all_conf.min().item()
            
            # Confidence when correct vs incorrect
            correct_conf = all_conf[all_correct].mean().item() if all_correct.any() else 0.0
            incorrect_conf = all_conf[~all_correct].mean().item() if (~all_correct).any() else 0.0
            
            # Count high-confidence predictions (>99%)
            high_conf_pct = (all_conf > 0.99).float().mean().item() * 100
            
            # Log calibration metrics
            self.log("val_conf_mean", mean_conf, prog_bar=True)
            self.log("val_conf_max", max_conf, prog_bar=False)
            self.log("val_conf_min", min_conf, prog_bar=False)
            self.log("val_conf_correct", correct_conf, prog_bar=False)
            self.log("val_conf_incorrect", incorrect_conf, prog_bar=False)
            self.log("val_high_conf_pct", high_conf_pct, prog_bar=False)
            
            # Print calibration summary
            print(f"\n[Calibration] mean={mean_conf:.3f}, max={max_conf:.3f}, "
                  f"correct={correct_conf:.3f}, incorrect={incorrect_conf:.3f}, "
                  f">99%: {high_conf_pct:.1f}%")
        
        # Reset tracking
        self._val_confidences = []
        self._val_corrects = []

    def test_step(self, batch, _):
        loss, preds, y, confidences = self._shared_step(batch)
        self.acc_te.update(preds, y)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_conf_mean", confidences.mean(), on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        acc = self.acc_te.compute()
        self.log("test_acc", acc, prog_bar=True)
        self.acc_te.reset()

    def on_train_epoch_start(self):
        if self.trainer and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", lr, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        name = self.optimizer_name
        if name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif name == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            raise ValueError("optimizer must be one of: adam | adamw | sgd | rmsprop")

        if self.scheduler_name == "cosine":
            if self.t_max is None:
                raise ValueError("When using scheduler='cosine', please set t_max=max_epochs")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=int(self.t_max), eta_min=self.eta_min
            )
            return [optimizer], [scheduler]

        return optimizer
