# mnist_datamodule.py

from typing import Optional
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
import lightning as L


class MNISTDataModule(L.LightningDataModule):

    """
    MNIST DataModule with optional light augmentation.
    """

    def __init__(
        self,
        data_dir: str = "./datasets",
        batch_size: int = 128,
        num_workers: int = 0,
        val_size: int = 5_000,
        augment: bool = False,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.augment = augment
        self.pin_memory = pin_memory

        self.train_transform = self._build_train_transform()
        self.eval_transform  = transforms.ToTensor()

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def _build_train_transform(self) -> transforms.Compose:
        if not self.augment:
            return transforms.Compose([transforms.ToTensor()])
        return transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
            transforms.ToTensor(),
        ])

    # Lightning hooks
    def prepare_data(self) -> None:
        MNIST(self.data_dir, train=True,  download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        full_train = MNIST(self.data_dir, train=True,  download=False, transform=self.train_transform)
        train_len = len(full_train) - self.val_size
        self.train_set, self.val_set = random_split(
            full_train, [train_len, self.val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.test_set  = MNIST(self.data_dir, train=False, download=False, transform=self.eval_transform)

    # Dataloaders
    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None, "Call setup() first"
        return DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0), drop_last=False
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_set is not None, "Call setup() first"
        return DataLoader(
            self.val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0), drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_set is not None, "Call setup() first"
        return DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0), drop_last=False
        )

    def predict_dataloader(self):
        # reuse test loader for predict()
        return self.test_dataloader()
