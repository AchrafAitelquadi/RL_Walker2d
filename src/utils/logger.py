"""
Training Logger pour enregistrer toutes les métriques pendant l'entraînement
"""

import pandas as pd
import os


class TrainingLogger:
    """
    Logger pour enregistrer toutes les métriques d'entraînement
    
    Enregistre:
    - Récompenses par épisode et évaluations
    - Pertes du critic et de l'actor
    - Tailles des buffers
    - Métriques EAS (archive, Q-filter, améliorations PSO)
    - Magnitudes des actions
    """
    def __init__(self):
        # Performance
        self.episode_rewards = []
        self.eval_rewards = []
        self.eval_timesteps = []
        self.eval_stds = []
        
        # Losses
        self.critic_losses = []
        self.actor_losses = []
        
        # Buffers
        self.buffer_sizes = []
        
        # EAS metrics
        self.archive_sizes = []
        self.q_filter_rates = []
        self.q_values_normal = []
        self.q_values_evolved = []
        self.pso_improvements = []
        
        # Exploration
        self.action_magnitudes = []
    
    def log_episode(self, reward):
        """Enregistre la récompense d'un épisode"""
        self.episode_rewards.append(reward)
    
    def log_eval(self, timestep, reward, std):
        """Enregistre les résultats d'évaluation"""
        self.eval_timesteps.append(timestep)
        self.eval_rewards.append(reward)
        self.eval_stds.append(std)
    
    def log_training(self, critic_loss, actor_loss, buffer_size):
        """Enregistre les métriques d'entraînement"""
        self.critic_losses.append(critic_loss)
        self.actor_losses.append(actor_loss)
        self.buffer_sizes.append(buffer_size)
    
    def log_eas(self, archive_size, q_filter_rate, q_normal, q_evolved, pso_improvement):
        """Enregistre les métriques EAS"""
        self.archive_sizes.append(archive_size)
        self.q_filter_rates.append(q_filter_rate)
        self.q_values_normal.append(q_normal)
        self.q_values_evolved.append(q_evolved)
        self.pso_improvements.append(pso_improvement)
    
    def log_action(self, action_magnitude):
        """Enregistre la magnitude d'une action"""
        self.action_magnitudes.append(action_magnitude)
    
    def save_to_csv(self, save_dir, algorithm_name):
        """
        Sauvegarde toutes les métriques en fichiers CSV
        
        Args:
            save_dir: Répertoire où sauvegarder les CSV
            algorithm_name: Nom de l'algorithme (TD3 ou EAS-TD3)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Episode rewards
        if self.episode_rewards:
            df_episodes = pd.DataFrame({
                'episode': range(1, len(self.episode_rewards) + 1),
                'reward': self.episode_rewards
            })
            df_episodes.to_csv(f'{save_dir}/{algorithm_name}_episode_rewards.csv', index=False)
        
        # 2. Evaluation metrics
        if self.eval_rewards:
            df_eval = pd.DataFrame({
                'timestep': self.eval_timesteps,
                'eval_reward': self.eval_rewards,
                'eval_std': self.eval_stds
            })
            df_eval.to_csv(f'{save_dir}/{algorithm_name}_evaluations.csv', index=False)
        
        # 3. Training losses
        if self.critic_losses:
            df_losses = pd.DataFrame({
                'step': range(1, len(self.critic_losses) + 1),
                'critic_loss': self.critic_losses,
                'actor_loss': self.actor_losses,
                'buffer_size': self.buffer_sizes
            })
            df_losses.to_csv(f'{save_dir}/{algorithm_name}_training_losses.csv', index=False)
        
        # 4. EAS metrics (if available)
        if self.archive_sizes:
            df_eas = pd.DataFrame({
                'step': range(1, len(self.archive_sizes) + 1),
                'archive_size': self.archive_sizes,
                'q_filter_rate': self.q_filter_rates,
                'q_value_normal': self.q_values_normal,
                'q_value_evolved': self.q_values_evolved,
                'pso_improvement': self.pso_improvements
            })
            df_eas.to_csv(f'{save_dir}/{algorithm_name}_eas_metrics.csv', index=False)
        
        # 5. Action magnitudes
        if self.action_magnitudes:
            df_actions = pd.DataFrame({
                'step': range(1, len(self.action_magnitudes) + 1),
                'action_magnitude': self.action_magnitudes
            })
            df_actions.to_csv(f'{save_dir}/{algorithm_name}_action_magnitudes.csv', index=False)
        
        print(f"  [OK] CSV files saved to {save_dir}/")
