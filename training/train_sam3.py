#!/usr/bin/env python3
"""
SAM3 Training Launcher Script

This script launches SAM3 fine-tuning with the specified configuration.

Usage:
    python train_sam3.py [--config CONFIG_PATH] [--num-gpus NUM_GPUS]

Example:
    python train_sam3.py --config configs/sam3_custom_finetune.yaml --num-gpus 1
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add SAM3 to path
SAM3_REPO = Path("/workspace/sam3").resolve()
if SAM3_REPO.exists():
    sys.path.insert(0, str(SAM3_REPO))
else:
    print(f"Error: SAM3 repository not found at {SAM3_REPO}")
    print("Please clone the repository first:")
    print("  git clone https://github.com/facebookresearch/sam3.git /workspace/sam3")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Launch SAM3 Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam3_custom_finetune.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=0,
        help="Whether to use cluster (SLURM) for training"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Training or validation mode"
    )
    return parser.parse_args()

def check_prerequisites():
    """Check if all prerequisites are met."""
    print("Checking prerequisites...")

    # Check SAM3 installation
    try:
        import sam3
        print("✓ SAM3 is installed")
    except ImportError:
        print("✗ SAM3 is not installed")
        print("  Please install: pip install -e /workspace/sam3[train,dev]")
        return False

    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            print("⚠️  CUDA not available, will use CPU (very slow)")
    except ImportError:
        print("✗ PyTorch not installed")
        return False

    return True

def launch_training(config_path: str, num_gpus: int, use_cluster: int, mode: str):
    """Launch SAM3 training."""
    train_script = SAM3_REPO / "sam3" / "train" / "train.py"

    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        print("Please ensure the SAM3 repository is properly cloned.")
        return False

    # Build command
    cmd = [
        sys.executable,
        str(train_script),
        "-c", config_path,
        "--use-cluster", str(use_cluster),
        "--num-gpus", str(num_gpus),
    ]

    if mode == "val":
        cmd.extend(["trainer.mode=val"])

    print(f"\nLaunching training with command:")
    print(" ".join(cmd))
    print("\n" + "="*70 + "\n")

    # Launch training
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "="*70)
        print("✓ Training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return False

def main():
    args = parse_args()

    print("SAM3 Training Launcher")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Mode: {args.mode}")
    print("="*70 + "\n")

    # Check prerequisites
    if not check_prerequisites():
        print("\nPrerequisites not met. Please fix the issues above.")
        sys.exit(1)

    # Launch training
    success = launch_training(
        config_path=args.config,
        num_gpus=args.num_gpus,
        use_cluster=args.use_cluster,
        mode=args.mode
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
