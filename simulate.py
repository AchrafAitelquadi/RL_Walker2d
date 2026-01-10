import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import argparse
import os
from collections import deque
import glob
from pathlib import Path


# ======================== NETWORKS (same as training) ========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=400):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 300)
        self.l3 = nn.Linear(300, 1)
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, 300)
        self.l6 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


# ======================== TD3 AGENT ========================
class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.max_action = max_action
    
    def select_action(self, state, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        if not deterministic:
            # Add small exploration noise
            noise = np.random.normal(0, 0.05 * self.max_action, size=action.shape)
            action = action + noise
            action = action.clip(-self.max_action, self.max_action)
        
        return action
    
    def load(self, filename):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        
        checkpoint = torch.load(filename, map_location='cpu')
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        print(f"✓ Model loaded from {filename}")
        print(f"  Training iterations: {checkpoint.get('total_it', 'N/A')}")


# ======================== UTILITY FUNCTIONS ========================
def find_models(search_dirs=['./models', './results', './kaggle', '.']):
    """
    Find all available model files (.pt)
    
    Args:
        search_dirs: List of directories to search
    
    Returns:
        Dictionary mapping model names to paths
    """
    models = {}
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        # Find all .pt files
        pattern = os.path.join(search_dir, '**', '*.pt')
        for model_path in glob.glob(pattern, recursive=True):
            model_name = os.path.basename(model_path)
            models[model_name] = model_path
    
    return models


def list_available_models():
    """
    Print all available trained models
    """
    models = find_models()
    
    if not models:
        print("No trained models found. Train a model first using run_experiment.py")
        return None
    
    print(f"\n{'='*70}")
    print(f"AVAILABLE MODELS")
    print(f"{'='*70}")
    for idx, (name, path) in enumerate(models.items(), 1):
        # Get file size
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{idx}. {name}")
        print(f"   Path: {path}")
        print(f"   Size: {size_mb:.2f} MB")
    print(f"{'='*70}\n")
    
    return models


# ======================== SIMULATION FUNCTIONS ========================
def simulate_episode(agent, env, render=True, deterministic=True, max_steps=1000):
    """
    Simulate a single episode
    
    Args:
        agent: Trained TD3 agent
        env: Gymnasium environment
        render: Whether to render the environment
        deterministic: Whether to use deterministic policy
        max_steps: Maximum steps per episode
    
    Returns:
        episode_reward: Total reward for the episode
        episode_length: Number of steps in the episode
        actions: List of actions taken
        states: List of states visited
    """
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    actions = []
    states = [state.copy()]
    
    for step in range(max_steps):
        # Select action
        action = agent.select_action(state, deterministic=deterministic)
        actions.append(action.copy())
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Accumulate reward
        episode_reward += reward
        episode_length += 1
        
        states.append(next_state.copy())
        state = next_state
        
        if done:
            break
    
    return episode_reward, episode_length, actions, states


def run_simulation(model_path, env_name="Walker2d-v4", num_episodes=10, 
                   render=True, deterministic=True, seed=0, save_video=False, max_steps=1000):
    """
    Run simulation with a trained agent
    
    Args:
        model_path: Path to the saved model
        env_name: Name of the environment
        num_episodes: Number of episodes to simulate
        render: Whether to render the environment
        deterministic: Whether to use deterministic policy
        seed: Random seed
        save_video: Whether to save video (requires render_mode='rgb_array')
        max_steps: Maximum steps per episode
    """
    # Validate model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found: {model_path}")
        print("\nSearching for available models...")
        models = list_available_models()
        if models:
            print("Please specify one of the above models using --model argument")
        return None, None
    
    # Create environment
    if save_video:
        env = gym.make(env_name, render_mode='rgb_array', max_episode_steps=max_steps)
        # Wrap with RecordVideo
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, video_folder='videos', 
                         episode_trigger=lambda x: True,
                         name_prefix=f'simulation_{env_name}')
    elif render:
        env = gym.make(env_name, render_mode='human', max_episode_steps=max_steps)
    else:
        env = gym.make(env_name, max_episode_steps=max_steps)
    
    # Set seed
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Create and load agent
    print(f"\n{'='*70}")
    print(f"SIMULATION SETUP")
    print(f"{'='*70}")
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Path: {model_path}")
    print(f"Number of episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Policy mode: {'Deterministic' if deterministic else 'Stochastic (with noise)'}")
    print(f"Rendering: {'Enabled' if render else 'Disabled'}")
    if save_video:
        print(f"Video recording: Enabled (saving to ./videos/)")
    print(f"{'='*70}\n")
    
    agent = TD3(state_dim, action_dim, max_action)
    agent.load(model_path)
    
    # Run episodes
    episode_rewards = []
    episode_lengths = []
    
    print(f"\n{'='*70}")
    print(f"RUNNING SIMULATION")
    print(f"{'='*70}\n")
    
    for episode in range(num_episodes):
        reward, length, actions, states = simulate_episode(
            agent, env, render=render, deterministic=deterministic, max_steps=max_steps
        )
        
        episode_rewards.append(reward)
        episode_lengths.append(length)
        
        # Calculate statistics
        action_magnitudes = [np.linalg.norm(a) for a in actions]
        avg_action = np.mean(action_magnitudes)
        max_action_mag = np.max(action_magnitudes)
        
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Length: {length} steps")
        print(f"  Action magnitude: {avg_action:.3f} (avg), {max_action_mag:.3f} (max)")
        print()
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"SIMULATION RESULTS")
    print(f"{'='*70}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"{'='*70}\n")
    
    env.close()
    
    return episode_rewards, episode_lengths


def evaluate_agent_performance(model_path, env_name="Walker2d-v4", 
                               num_episodes=100, seed=0, max_steps=1000):
    """
    Evaluate agent performance without rendering (faster)
    
    Args:
        model_path: Path to the saved model
        env_name: Name of the environment
        num_episodes: Number of episodes to evaluate
        seed: Random seed
        max_steps: Maximum steps per episode
    
    Returns:
        Statistics dictionary
    """
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found: {model_path}")
        list_available_models()
        return None
    
    env = gym.make(env_name, max_episode_steps=max_steps)
    env = gym.make(env_name)
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = TD3(state_dim, action_dim, max_action)
    agent.load(model_path)
    
    print(f"\nEvaluating agent over {num_episodes} episodes...")
    
    episode_rewards = [], max_steps=max_steps
    episode_lengths = []
    success_count = 0  # Count episodes above a threshold
    
    for episode in range(num_episodes):
        reward, length, _, _ = simulate_episode(
            agent, env, render=False, deterministic=True
        )
        episode_rewards.append(reward)
        episode_lengths.append(length)
        
        # Count successes (e.g., reward > 0 for some environments)
        if reward > 0:
            success_count += 1
        
        if (episode + 1) % 10 == 0:
            print(f"  Progress: {episode + 1}/{num_episodes} episodes")
    
    env.close()
    
    stats = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'median_reward': np.median(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / num_episodes,
        'all_rewards': episode_rewards
    }
    
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS ({num_episodes} episodes)")
    print(f"{'='*70}")
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Median Reward: {stats['median_reward']:.2f}")
    print(f"Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
    print(f"Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Success Rate: {stats['success_rate']:.2%} (reward > 0)")
    print(f"{'='*70}\n")
    
    return stats


def compare_deterministic_vs_stochastic(model_path, env_name="Walker2d-v4", 
                                        num_episodes=50, seed=0, max_steps=1000):
    """
    Compare performance with deterministic vs stochastic policy
    
    Args:
        model_path: Path to the saved model
        env_name: Name of the environment
        num_episodes: Number of episodes per policy type
        seed: Random seed
        max_steps: Maximum steps per episode
    
    Returns:
        Tuple of (deterministic_rewards, stochastic_rewards)
    """
    if not os.path.exists(model_path):
        print(f"\n❌ Model file not found: {model_path}")
        list_available_models()
        return None, None
    
    env = gym.make(env_name, max_episode_steps=max_steps)
    env.action_space.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    agent = TD3(state_dim, action_dim, max_action)
    agent.load(model_path)
    
    print(f"\n{'='*70}")
    print(f"COMPARING DETERMINISTIC VS STOCHASTIC POLICY")
    print(f"{'='*70}\n")
    
    # Deterministic
    print("Running with DETERMINISTIC policy...")
    det_rewards = []
    for _ in range(num_episodes):
        reward, _, _, _ = simulate_episode(agent, env, render=False, deterministic=True, max_steps=max_steps)
        det_rewards.append(reward)
    
    # Stochastic
    print("Running with STOCHASTIC policy...")
    sto_rewards = []
    for _ in range(num_episodes):
        reward, _, _, _ = simulate_episode(agent, env, render=False, deterministic=False, max_steps=max_steps)
        sto_rewards.append(reward)
    
    env.close()
    
    print(f"\n{'='*70}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"Deterministic Policy:")
    print(f"  Mean: {np.mean(det_rewards):.2f} ± {np.std(det_rewards):.2f}")
    print(f"Stochastic Policy:")
    print(f"  Mean: {np.mean(sto_rewards):.2f} ± {np.std(sto_rewards):.2f}")
    print(f"{'='*70}\n")
    
    return det_rewards, sto_rewards


# ======================== MAIN ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Simulate trained TD3/EAS-TD3 agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python simulate.py --list-models
  
  # Simulate with rendering (10 episodes)
  python simulate.py --model results/TD3_Walker2d.pt
  
  # Full evaluation (100 episodes, no render)
  python simulate.py --model results/EAS_TD3_Walker2d.pt --evaluate
  
  # Compare deterministic vs stochastic
  python simulate.py --model results/TD3_Walker2d.pt --compare
  
  # Save video
  python simulate.py --model results/TD3_Walker2d.pt --episodes 5 --save-video
        """
    )
    
    parser.add_argument('--model', type=str, default=None,
                       help='Path to the saved model (.pt file)')
    parser.add_argument('--env', type=str, default='Walker2d-v4',
                       help='Environment name (default: Walker2d-v4)')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to simulate (default: 10)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed (default: 0)')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable rendering')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy instead of deterministic')
    parser.add_argument('--list-models', action='store_true',
                       help='List all available trained models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run full evaluation (100 episodes, no render)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare deterministic vs stochastic policies')
    parser.add_argument('--save-video', action='store_true',
                       help='Save video of episodes')
    
    args = parser.parse_args()
    
    # List models mode
    if args.list_models:
        list_available_models()
        exit(0)
    
    # Auto-detect model if not specified
    if args.model is None:
        print("No model specified. Searching for available models...")
        models = find_models()
        
        if not models:
            print("\n❌ No trained models found.")
            print("Train a model first using: python src/run_experiment.py")
            exit(1)
        
        # Use the first model found
        model_name, model_path = list(models.items())[0]
        print(f"\n✓ Auto-selected model: {model_name}")
        print(f"  Path: {model_path}\n")
        args.model = model_path
    
    # Mode selection
    if args.evaluate:
        # Full evaluation mode
        evaluate_agent_performance(
            model_path=args.model,
            env_name=args.env,
            num_episodes=100,
            seed=args.seed,
            max_steps=args.max_steps
        )
    elif args.compare:
        # Comparison mode
        compare_deterministic_vs_stochastic(
            model_path=args.model,
            env_name=args.env,
            num_episodes=50,
            seed=args.seed,
            max_steps=args.max_steps
        )
    else:
        # Standard simulation mode
        run_simulation(
            model_path=args.model,
            env_name=args.env,
            num_episodes=args.episodes,
            render=not args.no_render,
            deterministic=not args.stochastic,
            seed=args.seed,
            save_video=args.save_video,
            max_steps=args.max_steps
        )