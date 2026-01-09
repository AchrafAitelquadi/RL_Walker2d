"""
Evolutionary Action Selection (EAS) using Particle Swarm Optimization
"""

import numpy as np
import torch


class EAS:
    """
    Evolutionary Action Selection via Particle Swarm Optimization (PSO)
    
    Optimise les actions en utilisant PSO pour maximiser leur Q-value
    
    Args:
        critic: Réseau Critic pour évaluer les actions
        action_dim (int): Dimension de l'espace d'action
        max_action (float): Valeur maximale des actions
        pop_size (int): Taille de la population PSO
        iterations (int): Nombre d'itérations PSO
        omega (float): Coefficient d'inertie
        c1 (float): Coefficient cognitif (personal best)
        c2 (float): Coefficient social (global best)
        vmax (float): Vélocité maximale (en proportion de max_action)
    """
    def __init__(self, critic, action_dim, max_action, 
                 pop_size=10, iterations=10, omega=1.2, c1=1.5, c2=1.5, vmax=0.1):
        self.critic = critic
        self.action_dim = action_dim
        self.max_action = max_action
        self.pop_size = pop_size
        self.iterations = iterations
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.vmax = vmax * max_action
    
    def evolve_action(self, state, action):
        """
        Évolue une action en utilisant PSO pour maximiser sa Q-value
        
        Args:
            state (np.ndarray): État actuel
            action (np.ndarray): Action initiale de la politique
            
        Returns:
            tuple: (action évoluée, amélioration de Q-value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Initialiser population avec bruit gaussien autour de l'action initiale
        population = []
        for _ in range(self.pop_size):
            noise = np.random.normal(0, 0.1, self.action_dim)
            noisy_action = np.clip(action + noise, -self.max_action, self.max_action)
            population.append(noisy_action)
        population = np.array(population)
        
        # Initialiser vélocités aléatoires
        velocities = np.random.uniform(-self.vmax, self.vmax, 
                                      (self.pop_size, self.action_dim))
        
        # Initialiser personal best et global best
        personal_best = population.copy()
        personal_best_scores = np.full(self.pop_size, -np.inf)
        global_best = action.copy()
        global_best_score = -np.inf
        
        # Q-value de l'action originale (baseline)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        with torch.no_grad():
            original_q = self.critic.Q1(state_tensor, action_tensor).item()
        
        # Évaluer population initiale
        for i in range(self.pop_size):
            action_tensor = torch.FloatTensor(population[i]).unsqueeze(0)
            with torch.no_grad():
                q_value = self.critic.Q1(state_tensor, action_tensor).item()
            personal_best_scores[i] = q_value
            if q_value > global_best_score:
                global_best_score = q_value
                global_best = population[i].copy()
        
        # Évolution PSO sur plusieurs itérations
        for t in range(self.iterations):
            for i in range(self.pop_size):
                # Évaluer fitness de la particule actuelle
                action_tensor = torch.FloatTensor(population[i]).unsqueeze(0)
                with torch.no_grad():
                    q_value = self.critic.Q1(state_tensor, action_tensor).item()
                
                # Mettre à jour personal best si meilleure fitness
                if q_value > personal_best_scores[i]:
                    personal_best_scores[i] = q_value
                    personal_best[i] = population[i].copy()
                    
                    # Mettre à jour global best si meilleure fitness globale
                    if q_value > global_best_score:
                        global_best_score = q_value
                        global_best = population[i].copy()
                
                # Mettre à jour vélocité selon équation PSO
                # v = ω*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (self.omega * velocities[i] + 
                               self.c1 * r1 * (personal_best[i] - population[i]) +
                               self.c2 * r2 * (global_best - population[i]))
                velocities[i] = np.clip(velocities[i], -self.vmax, self.vmax)
                
                # Mettre à jour position
                population[i] = population[i] + velocities[i]
                population[i] = np.clip(population[i], -self.max_action, self.max_action)
        
        # Retourner la meilleure action trouvée et son amélioration
        improvement = global_best_score - original_q
        return global_best, improvement
