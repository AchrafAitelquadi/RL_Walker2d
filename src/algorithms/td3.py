"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) avec EAS
"""

import torch
import torch.nn.functional as F
from .networks import Actor, Critic
from .eas import EAS
from .replay_buffer import Archive


class TD3:
    """
    Algorithme TD3 (Twin Delayed DDPG)
    
    Améliore DDPG avec:
    - Double Q-Learning (deux critiques)
    - Delayed Policy Updates (mise à jour retardée de l'actor)
    - Target Policy Smoothing (bruit sur les actions cibles)
    
    Optionnel: Evolutionary Action Selection (EAS)
    
    Args:
        state_dim (int): Dimension de l'espace d'état
        action_dim (int): Dimension de l'espace d'action
        max_action (float): Valeur maximale des actions
        use_eas (bool): Activer EAS pour améliorer l'apprentissage
    """
    def __init__(self, state_dim, action_dim, max_action, use_eas=False):
        # Actor et son target
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Critic et son target
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.action_dim = action_dim
        self.use_eas = use_eas
        
        # EAS optionnel
        if use_eas:
            self.eas = EAS(self.critic, action_dim, max_action)
            self.archive = Archive(max_size=100000)
        
        self.total_it = 0
    
    def select_action(self, state):
        """
        Sélectionne une action déterministe basée sur l'état
        
        Args:
            state (np.ndarray): État actuel
            
        Returns:
            np.ndarray: Action sélectionnée
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, batch_size=100, discount=0.99, tau=0.005,
              policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """
        Effectue une étape d'entraînement TD3
        
        Args:
            replay_buffer: Buffer de replay pour échantillonner
            batch_size (int): Taille du batch
            discount (float): Facteur de discount γ
            tau (float): Coefficient de soft update
            policy_noise (float): Écart-type du bruit pour target policy
            noise_clip (float): Clipping du bruit
            policy_freq (int): Fréquence de mise à jour de la politique
            
        Returns:
            tuple: (critic_loss, actor_loss, q_filter_rate, q_normal, q_evolved)
        """
        self.total_it += 1
        
        # Échantillonner un batch de transitions
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        with torch.no_grad():
            # Target Policy Smoothing: ajouter du bruit aux actions cibles
            noise = (torch.randn_like(actions) * policy_noise).clamp(-noise_clip, noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, 
                                                                          self.max_action)
            
            # Clipped Double Q-Learning: prendre le min des deux Q-values cibles
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * discount * target_q
        
        # Calculer Q-values actuelles
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic loss: MSE entre Q actuelles et Q cibles
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimiser Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss_value = None
        q_filter_rate = 0
        q_normal_mean = 0
        q_evolved_mean = 0
        
        # Delayed Policy Updates: mettre à jour l'actor moins fréquemment
        if self.total_it % policy_freq == 0:
            # Actor loss: Gradient de politique (maximiser Q)
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # EAS: Gradient d'action évolutionnaire
            if self.use_eas and self.archive.size() >= batch_size:
                evo_states, evo_actions = self.archive.sample(batch_size)
                if evo_states is not None:
                    evo_states = torch.FloatTensor(evo_states)
                    evo_actions = torch.FloatTensor(evo_actions)
                    
                    # Q-Filter: ne garder que les actions évoluées meilleures
                    current_actions = self.actor(evo_states)
                    with torch.no_grad():
                        q_current = self.critic.Q1(evo_states, current_actions)
                        q_evo = self.critic.Q1(evo_states, evo_actions)
                        q_filter = (q_evo > q_current).float()
                        
                        # Statistiques pour logging
                        q_filter_rate = q_filter.mean().item()
                        q_normal_mean = q_current.mean().item()
                        q_evolved_mean = q_evo.mean().item()
                    
                    # Loss évolutionnaire: apprendre à imiter les bonnes actions
                    evo_loss = q_filter * F.mse_loss(self.actor(evo_states), 
                                                      evo_actions, reduction='none').mean(dim=1, keepdim=True)
                    evo_loss = evo_loss.mean()
                    
                    actor_loss = actor_loss + evo_loss
            
            actor_loss_value = actor_loss.item()
            
            # Optimiser Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update des réseaux cibles: θ' ← τθ + (1-τ)θ'
            for param, target_param in zip(self.critic.parameters(), 
                                          self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), 
                                          self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        return critic_loss.item(), actor_loss_value, q_filter_rate, q_normal_mean, q_evolved_mean
    
    def save(self, filename):
        """
        Sauvegarde le modèle complet
        
        Args:
            filename (str): Chemin du fichier de sauvegarde
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_it': self.total_it,
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        """
        Charge un modèle sauvegardé
        
        Args:
            filename (str): Chemin du fichier à charger
        """
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.total_it = checkpoint['total_it']
        print(f"Model loaded from {filename}")
