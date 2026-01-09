"""
Neural Networks: Actor and Critic pour TD3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Réseau Actor (Politique déterministe)
    
    Transforme un état en action continue
    Architecture: état (17) → 400 → 300 → action (6)
    
    Args:
        state_dim (int): Dimension de l'espace d'état
        action_dim (int): Dimension de l'espace d'action
        max_action (float): Valeur maximale des actions
        hidden_dim (int): Taille de la première couche cachée
    """
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=400):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 300)
        self.l3 = nn.Linear(300, action_dim)
        self.max_action = max_action
    
    def forward(self, state):
        """
        Forward pass
        
        Args:
            state (torch.Tensor): État(s) d'entrée
            
        Returns:
            torch.Tensor: Action(s) dans [-max_action, max_action]
        """
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    """
    Réseau Critic (Double Q-Learning)
    
    Estime la valeur Q(s,a) avec deux réseaux indépendants
    Architecture: (état + action) (23) → 400 → 300 → Q-value (1)
    
    Args:
        state_dim (int): Dimension de l'espace d'état
        action_dim (int): Dimension de l'espace d'action
        hidden_dim (int): Taille de la première couche cachée
    """
    def __init__(self, state_dim, action_dim, hidden_dim=400):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 300)
        self.l3 = nn.Linear(300, 1)
        
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l5 = nn.Linear(hidden_dim, 300)
        self.l6 = nn.Linear(300, 1)
    
    def forward(self, state, action):
        """
        Forward pass pour les deux Q-networks
        
        Args:
            state (torch.Tensor): État(s)
            action (torch.Tensor): Action(s)
            
        Returns:
            tuple: (Q1-value, Q2-value)
        """
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """
        Calcule uniquement Q1 (utilisé pour la loss de l'actor)
        
        Args:
            state (torch.Tensor): État(s)
            action (torch.Tensor): Action(s)
            
        Returns:
            torch.Tensor: Q1-value
        """
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
