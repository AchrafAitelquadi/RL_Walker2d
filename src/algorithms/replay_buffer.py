"""
Replay Buffer and Archive for experience storage
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Replay Buffer pour stocker les transitions (s, a, r, s', done)
    
    Args:
        max_size (int): Capacité maximale du buffer
    """
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        """Ajoute une transition au buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Échantillonne un batch aléatoire de transitions
        
        Args:
            batch_size (int): Taille du batch
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def size(self):
        """Retourne la taille actuelle du buffer"""
        return len(self.buffer)


class Archive:
    """
    Archive pour stocker les paires (état, action évoluée) pour EAS
    
    Args:
        max_size (int): Capacité maximale de l'archive
    """
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action):
        """Ajoute une paire (état, action évoluée) à l'archive"""
        self.buffer.append((state, action))
    
    def sample(self, batch_size):
        """
        Échantillonne un batch aléatoire de paires (état, action)
        
        Args:
            batch_size (int): Taille du batch
            
        Returns:
            tuple: (states, actions) ou (None, None) si insuffisant
        """
        if len(self.buffer) < batch_size:
            return None, None
        batch = random.sample(self.buffer, batch_size)
        states, actions = zip(*batch)
        return np.array(states), np.array(actions)
    
    def size(self):
        """Retourne la taille actuelle de l'archive"""
        return len(self.buffer)
