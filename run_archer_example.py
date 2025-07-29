#!/usr/bin/env python3
"""
Example script to run RAGEN with ArCHer baseline
ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL
by Yifei Zhou et al. (ICML 2024)
"""

import subprocess
import sys

def run_archer_training():
    """Run RAGEN training with ArCHer baseline"""
    cmd = [
        "python", "train.py",
        "--config-name", "archer",
        "trainer.total_training_steps=50",  # Short training for demo
        "trainer.experiment_name=archer_demo"
    ]
    
    print("Running ArCHer baseline training...")
    print("Command:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

if __name__ == "__main__":
    success = run_archer_training()
    sys.exit(0 if success else 1)