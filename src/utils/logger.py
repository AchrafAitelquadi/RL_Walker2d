"""
Training Logger pour enregistrer toutes les métriques pendant l'entraînement
"""


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
