#!/usr/bin/env python3
"""
Test script for ArCHer integration in RAGEN.
This script verifies that ArCHer can be loaded and run end-to-end.
"""

import os
import sys
import torch
import hydra
from omegaconf import DictConfig

# Add RAGEN to path
sys.path.append('/root/RAGEN')

from ragen.trainer.agent_trainer import RayAgentTrainer
from ragen.utils import register_resolvers

def test_archer_config_loading():
    """Test that ArCHer configuration loads correctly"""
    print("Testing ArCHer configuration loading...")
    
    # Register resolvers
    register_resolvers()
    
    # Load ArCHer config
    with hydra.initialize(version_base=None, config_path="config"):
        cfg = hydra.compose(config_name="archer")
    
    print(f"✓ Config loaded successfully")
    print(f"  - Algorithm: {cfg.algorithm.adv_estimator}")
    print(f"  - ArCHer enabled: {cfg.archer.enabled}")
    print(f"  - ArCHer tau: {cfg.archer.tau}")
    print(f"  - ArCHer alpha: {cfg.archer.alpha}")
    print(f"  - Critic epochs: {cfg.critic.ppo_epochs}")
    print(f"  - Actor epochs: {cfg.actor_rollout_ref.actor.ppo_epochs}")
    
    return cfg

def test_archer_imports():
    """Test that ArCHer classes can be imported"""
    print("\nTesting ArCHer imports...")
    
    try:
        from ragen.workers.critic.archer_critic import ArCherCritic, ArCherDoubleCriticModule
        print("✓ ArCHer critic classes imported successfully")
        
        from ragen.trainer.core_algos import compute_archer_advantage_return
        print("✓ ArCHer advantage computation imported successfully")
        
        from verl.trainer.ppo.ray_trainer import AdvantageEstimator
        assert hasattr(AdvantageEstimator, 'ARCHER'), "ARCHER enum not found"
        print("✓ ArCHer advantage estimator enum found")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    return True

def test_archer_critic_initialization():
    """Test that ArCHer critic can be initialized"""
    print("\nTesting ArCHer critic initialization...")
    
    try:
        from ragen.workers.critic.archer_critic import ArCherDoubleCriticModule
        from transformers import AutoModel
        
        # Create a mock base critic module
        base_model = AutoModel.from_pretrained("gpt2", num_labels=1)
        
        # Create ArCHer critic module
        archer_module = ArCherDoubleCriticModule(base_model)
        print("✓ ArCHer Double Critic Module created successfully")
        
        # Test forward pass with dummy data
        batch_size, seq_len, hidden_size = 2, 10, base_model.config.hidden_size
        
        # Test observation/action encoding
        obs_ids = torch.randint(0, 1000, (batch_size, seq_len))
        action_ids = torch.randint(0, 1000, (batch_size, seq_len))
        obs_mask = torch.ones(batch_size, seq_len)
        action_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = archer_module(
                observation_ids=obs_ids,
                observation_mask=obs_mask, 
                action_ids=action_ids,
                action_mask=action_mask
            )
        
        # Check outputs
        assert hasattr(output, 'q1_values'), "Missing q1_values"
        assert hasattr(output, 'v1_values'), "Missing v1_values"
        print(f"✓ Forward pass successful, output shape: {output.q1_values.shape}")
        
    except Exception as e:
        print(f"✗ ArCHer critic initialization failed: {e}")
        return False
    
    return True

def test_archer_advantage_computation():
    """Test ArCHer advantage computation"""
    print("\nTesting ArCHer advantage computation...")
    
    try:
        from ragen.trainer.core_algos import compute_archer_advantage_return
        
        # Create dummy Q and V values
        batch_size, seq_len = 4, 20
        q_values = torch.randn(batch_size, seq_len)
        v_values = torch.randn(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)
        gamma = 0.95
        
        # Compute advantages
        advantages, returns = compute_archer_advantage_return(
            q_values=q_values,
            v_values=v_values,
            loss_mask=loss_mask,
            gamma=gamma
        )
        
        assert advantages.shape == (batch_size, seq_len), f"Wrong advantage shape: {advantages.shape}"
        assert returns.shape == (batch_size, seq_len), f"Wrong returns shape: {returns.shape}"
        
        print(f"✓ Advantage computation successful")
        print(f"  - Advantage mean: {advantages.mean().item():.4f}")
        print(f"  - Returns mean: {returns.mean().item():.4f}")
        
    except Exception as e:
        print(f"✗ ArCHer advantage computation failed: {e}")
        return False
    
    return True

def main():
    """Run all ArCHer tests"""
    print("=" * 60)
    print("RAGEN ArCHer Integration Test Suite")
    print("=" * 60)
    
    tests = [
        test_archer_imports,
        test_archer_config_loading,
        test_archer_critic_initialization,
        test_archer_advantage_computation,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"ArCHer Integration Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! ArCHer integration is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
