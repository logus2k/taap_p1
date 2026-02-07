#!/usr/bin/env python3
"""
Train calibrated MNIST CNN classifier.

Usage:
    python train_calibrated_classifier.py
    python train_calibrated_classifier.py --label-smoothing 0.1 --dropout 0.2 --epochs 25
"""

import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from mnist_cnn_classifier_calibrated import MNISTCNNCalibrated
from mnist_datamodule import MNISTDataModule


def main():
    parser = argparse.ArgumentParser(description="Train calibrated MNIST classifier")
    
    # Model hyperparameters
    parser.add_argument("--width", type=int, default=128, help="Base conv channels")
    parser.add_argument("--depth", type=int, default=3, help="Number of conv blocks (1-3)")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--use-bn", action="store_true", default=False, help="Use batch normalization")
    parser.add_argument("--weight-init", type=str, default="he", help="Weight initialization (he/xavier)")
    
    # Calibration
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    
    # Training
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=25, help="Max epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    
    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine", help="LR scheduler (none/cosine)")
    parser.add_argument("--eta-min", type=float, default=1e-5, help="Min LR for cosine scheduler")
    
    # Paths
    parser.add_argument("--data-dir", type=str, default="./datasets", help="Dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--exp-name", type=str, default="mnist_cnn_calibrated", help="Experiment name")
    
    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    L.seed_everything(args.seed)
    
    # Set precision for faster training on supported GPUs
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    
    # Print configuration
    print("\n" + "=" * 60)
    print("CALIBRATED MNIST CLASSIFIER TRAINING")
    print("=" * 60)
    print(f"\nCalibration settings:")
    print(f"  label_smoothing: {args.label_smoothing}")
    print(f"  dropout_p: {args.dropout}")
    print(f"\nModel settings:")
    print(f"  width: {args.width}")
    print(f"  depth: {args.depth}")
    print(f"  activation: {args.activation}")
    print(f"  use_bn: {args.use_bn}")
    print(f"  weight_init: {args.weight_init}")
    print(f"\nTraining settings:")
    print(f"  lr: {args.lr}")
    print(f"  optimizer: {args.optimizer}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  epochs: {args.epochs}")
    print(f"  scheduler: {args.scheduler}")
    print("=" * 60 + "\n")
    
    # Data
    dm = MNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        augment=True,
    )
    
    # Model
    model = MNISTCNNCalibrated(
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        activation=args.activation,
        width=args.width,
        depth=args.depth,
        dropout_p=args.dropout,
        use_bn=args.use_bn,
        weight_init=args.weight_init,
        label_smoothing=args.label_smoothing,
        scheduler=args.scheduler,
        t_max=args.epochs,
        eta_min=args.eta_min,
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=args.patience,
            min_delta=1e-4,
        ),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            dirpath=args.checkpoint_dir,
            filename="mnist_cnn_calibrated_best",
        ),
    ]
    
    # Logger
    logger = CSVLogger(
        save_dir=args.log_dir,
        name=args.exp_name,
    )
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        precision="32-true",
        log_every_n_steps=10,
        deterministic=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        logger=logger,
    )
    
    # Train
    trainer.fit(model, datamodule=dm)
    
    # Test
    print("\n" + "=" * 60)
    print("TESTING BEST MODEL")
    print("=" * 60)
    trainer.test(ckpt_path="best", datamodule=dm)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nCheckpoint saved to: {args.checkpoint_dir}/mnist_cnn_calibrated_best.ckpt")
    print(f"Logs saved to: {args.log_dir}/{args.exp_name}/")
    print("\nCalibration targets:")
    print("  - val_conf_mean: Should be ~85-95% (not 99%+)")
    print("  - val_high_conf_pct: Should be low (<10%)")
    print("  - val_conf_incorrect: Should be lower than val_conf_correct")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
