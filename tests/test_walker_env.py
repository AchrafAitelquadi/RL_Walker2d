"""
Test script for Walker2d-v4 environment

Run tests to verify the environment is working correctly.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environments import make_walker_env, get_env_info


def test_basic_functionality():
    """
    Test basic environment creation
    """
    print("\n" + "="*70)
    print("TEST 1: Environment Creation")
    print("="*70)
    
    env = make_walker_env(seed=42)
    info = get_env_info(env)
    
    print(f"Name: {info['name']}")
    print(f"State dim: {info['state_dim']}")
    print(f"Action dim: {info['action_dim']}")
    print(f"Action range: [{info['action_min']}, {info['action_max']}]")
    print(f"Max steps: {info['max_episode_steps']}")
    
    env.close()
    print("\n[OK] Test passed!\n")


def test_random_policy():
    """
    Test random actions
    """
    print("\n" + "="*70)
    print("TEST 2: Random Policy")
    print("="*70)
    
    env = make_walker_env(seed=42)
    rewards = []
    
    for ep in range(3):
        state, _ = env.reset(seed=42 + ep)
        total_reward = 0
        
        for _ in range(100):
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
        print(f"Episode {ep+1}: {total_reward:.2f}")
    
    env.close()
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print("\n[OK] Test passed!\n")


def test_rendering():
    """
    Test with visual rendering
    """
    print("\n" + "="*70)
    print("TEST 3: Rendering")
    print("="*70)
    
    env = make_walker_env(render_mode='human', max_episode_steps=1000, seed=42)
    
    num_episodes = 3
    print(f"Running {num_episodes} episodes with rendering...")
    print("Watch the walker learn to move!\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset(seed=42 + episode)
        total_reward = 0
        steps = 0
        
        for _ in range(1000):
            action = env.action_space.sample()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode+1}/{num_episodes}: Reward={total_reward:.2f}, Steps={steps}")
    
    env.close()
    print("\n[OK] Test passed!\n")


def run_all_tests(include_rendering=False):
    """
    Run all tests
    """
    print("\n" + "="*70)
    print("WALKER2D-V4 ENVIRONMENT TESTS")
    print("="*70)
    
    try:
        test_basic_functionality()
        test_random_policy()
        
        if include_rendering:
            test_rendering()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Walker2d-v4 environment')
    parser.add_argument('--render', action='store_true', help='Include rendering test')
    
    args = parser.parse_args()
    
    run_all_tests(include_rendering=args.render)
