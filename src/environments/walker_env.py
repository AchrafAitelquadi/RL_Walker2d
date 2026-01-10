"""
Walker2d-v4 Environment Creation

This module provides the essential function to create the Walker2d environment.
"""

import gymnasium as gym
from typing import Optional, Dict, Any


def make_walker_env(
    env_name: str = "Walker2d-v4",
    render_mode: Optional[str] = None,
    max_episode_steps: int = 1000,
    seed: Optional[int] = None
) -> gym.Env:
    """
    Create a Walker2d environment
    
    Args:
        env_name: Name of the environment (default: Walker2d-v4)
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        max_episode_steps: Maximum steps per episode
        seed: Random seed for reproducibility
    
    Returns:
        Configured Gymnasium environment
    """
    env = gym.make(
        env_name,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps
    )
    
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    
    return env


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """
    Get environment specifications
    
    Args:
        env: Gymnasium environment
    
    Returns:
        Dictionary with environment info
    """
    return {
        'name': env.spec.id if env.spec else 'Unknown',
        'state_dim': env.observation_space.shape[0],
        'action_dim': env.action_space.shape[0],
        'action_min': float(env.action_space.low[0]),
        'action_max': float(env.action_space.high[0]),
        'max_episode_steps': env.spec.max_episode_steps if env.spec else None,
    }
