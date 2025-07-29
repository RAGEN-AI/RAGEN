#!/usr/bin/env python3
"""
Simple test script to identify the root cause of alfworld issues.
"""

import os
import sys
sys.path.append('/root/RAGEN')

def test_alfworld_environment():
    print("=== Testing Alfworld Environment ===")
    
    # Test 1: Check environment variables
    print("\n1. Checking environment variables:")
    alfworld_data = os.environ.get('ALFWORLD_DATA')
    print(f"   ALFWORLD_DATA = {alfworld_data}")
    
    if alfworld_data and os.path.exists(alfworld_data):
        print(f"   ✓ ALFWORLD_DATA path exists")
        # Check specific subdirectories
        train_path = os.path.join(alfworld_data, 'json_2.1.1', 'train')
        print(f"   Train data path: {train_path}")
        print(f"   Train data exists: {os.path.exists(train_path)}")
        if os.path.exists(train_path):
            files = os.listdir(train_path)
            print(f"   Number of files in train: {len(files)}")
    else:
        print(f"   ✗ ALFWORLD_DATA path does not exist or not set")
    
    # Test 2: Check imports
    print("\n2. Checking imports:")
    try:
        import textworld
        print("   ✓ textworld imported successfully")
    except ImportError as e:
        print(f"   ✗ textworld import failed: {e}")
        return
    
    try:
        import alfworld
        print("   ✓ alfworld imported successfully")
    except ImportError as e:
        print(f"   ✗ alfworld import failed: {e}")
        return
    
    # Test 3: Try to create alfworld environment
    print("\n3. Testing alfworld environment creation:")
    try:
        from ragen.env.alfworld_old.config import AlfredEnvConfig
        from ragen.env.alfworld_old.env import AlfredTXTEnv
        
        config = AlfredEnvConfig()
        print(f"   Config file: {config.config_file}")
        
        # Test if config file exists
        if os.path.exists(config.config_file):
            print("   ✓ Config file exists")
        else:
            print(f"   ✗ Config file does not exist: {config.config_file}")
            return
        
        env = AlfredTXTEnv(config)
        print(f"   ✓ Environment created successfully")
        print(f"   Number of games: {env.num_games}")
        print(f"   Number of game files: {len(env.game_files) if hasattr(env, 'game_files') else 'N/A'}")
        
        # Test reset
        print("\n4. Testing environment reset:")
        obs = env.reset(seed=42)
        print(f"   ✓ Reset successful")
        print(f"   Observation type: {type(obs)}")
        print(f"   Observation length: {len(str(obs)) if obs else 0}")
        print(f"   Observation preview: {str(obs)[:200]}..." if obs else "Empty observation")
        
        # Test step
        print("\n5. Testing environment step:")
        action = "look"  # Simple action
        obs, reward, done, info = env.step(action)
        print(f"   ✓ Step successful")
        print(f"   Observation length: {len(str(obs)) if obs else 0}")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        print(f"   Info: {info}")
        
    except Exception as e:
        print(f"   ✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_alfworld_environment()
