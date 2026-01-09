"""
Visualisation avancée des résultats d'entraînement
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

# Configuration pour de beaux graphiques
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


class EnhancedPlotter:
    """
    Classe pour créer des visualisations avancées des résultats
    
    Args:
        save_dir (str): Dossier où sauvegarder les graphiques
    """
    def __init__(self, save_dir="results/figures"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def smooth_curve(self, data, window=50):
        """
        Lisse une courbe avec une moyenne mobile
        
        Args:
            data: Données à lisser
            window (int): Taille de la fenêtre
            
        Returns:
            np.ndarray: Données lissées
        """
        if len(data) < window:
            return data
        return uniform_filter1d(data, size=window, mode='nearest')
    
    def plot_performance_analysis(self, logger, model_name):
        """
        Analyse complète des performances
        
        Args:
            logger: TrainingLogger avec les métriques
            model_name (str): Nom du modèle
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Récompenses par épisode avec moyenne mobile
        ax = axes[0, 0]
        episodes = np.arange(len(logger.episode_rewards))
        ax.plot(episodes, logger.episode_rewards, alpha=0.3, 
                color='steelblue', label='Episode Reward')
        
        # Moyennes mobiles avec différentes fenêtres
        for window in [50, 100]:
            if len(logger.episode_rewards) > window:
                smoothed = self.smooth_curve(logger.episode_rewards, window)
                ax.plot(episodes, smoothed, linewidth=2, 
                       label=f'MA-{window}')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards with Moving Averages')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Évaluations avec intervalle de confiance
        ax = axes[0, 1]
        timesteps = np.array(logger.eval_timesteps)
        rewards = np.array(logger.eval_rewards)
        stds = np.array(logger.eval_stds)
        
        ax.plot(timesteps, rewards, linewidth=2.5, 
                color='darkgreen', marker='o', markersize=4)
        ax.fill_between(timesteps, 
                        rewards - stds,
                        rewards + stds,
                        alpha=0.3, color='darkgreen')
        ax.fill_between(timesteps, 
                        rewards - 2*stds,
                        rewards + 2*stds,
                        alpha=0.15, color='darkgreen')
        
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Average Reward')
        ax.set_title('Evaluation Performance (±1σ, ±2σ)')
        ax.grid(True, alpha=0.3)
        
        # 3. Stabilité (écart-type des évaluations)
        ax = axes[1, 0]
        ax.plot(timesteps, stds, linewidth=2, 
                color='coral', marker='s', markersize=4)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Performance Stability (Lower is Better)')
        ax.grid(True, alpha=0.3)
        
        # 4. Distribution des récompenses par période
        ax = axes[1, 1]
        if len(logger.episode_rewards) > 100:
            # Diviser en quartiles
            n = len(logger.episode_rewards)
            quartiles = [
                logger.episode_rewards[:n//4],
                logger.episode_rewards[n//4:n//2],
                logger.episode_rewards[n//2:3*n//4],
                logger.episode_rewards[3*n//4:]
            ]
            labels = ['Q1 (Early)', 'Q2', 'Q3', 'Q4 (Late)']
            
            bp = ax.boxplot(quartiles, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], 
                                   ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Reward')
            ax.set_title('Reward Distribution by Training Phase')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{model_name}_performance_analysis.png')
        plt.close()
    
    def plot_learning_curves(self, logger, model_name):
        """
        Courbes d'apprentissage détaillées
        
        Args:
            logger: TrainingLogger avec les métriques
            model_name (str): Nom du modèle
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Critic Loss
        ax = axes[0, 0]
        if logger.critic_losses:
            steps = np.arange(len(logger.critic_losses))
            ax.plot(steps, logger.critic_losses, alpha=0.4, color='blue')
            smoothed = self.smooth_curve(logger.critic_losses, window=100)
            ax.plot(steps, smoothed, linewidth=2, color='darkblue', 
                   label='Smoothed')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Critic Loss')
            ax.set_title('Critic Loss Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # 2. Actor Loss
        ax = axes[0, 1]
        actor_losses = [x for x in logger.actor_losses if x is not None]
        if actor_losses:
            steps = np.arange(len(actor_losses))
            ax.plot(steps, actor_losses, alpha=0.4, color='orange')
            smoothed = self.smooth_curve(actor_losses, window=100)
            ax.plot(steps, smoothed, linewidth=2, color='darkorange', 
                   label='Smoothed')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Actor Loss (Negative Q-value)')
            ax.set_title('Actor Loss Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Replay Buffer Size
        ax = axes[1, 0]
        if logger.buffer_sizes:
            ax.plot(logger.buffer_sizes, linewidth=2, color='purple')
            ax.axhline(y=max(logger.buffer_sizes), 
                      color='red', linestyle='--', alpha=0.5,
                      label=f'Max: {max(logger.buffer_sizes):,}')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Buffer Size')
            ax.set_title('Replay Buffer Growth')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Loss Ratio (indicateur de stabilité)
        ax = axes[1, 1]
        if logger.critic_losses and len(logger.critic_losses) > 100:
            window = 100
            early_loss = np.mean(logger.critic_losses[:window])
            loss_ratio = [np.mean(logger.critic_losses[max(0, i-window):i+1]) / early_loss 
                         for i in range(window, len(logger.critic_losses))]
            ax.plot(range(window, len(logger.critic_losses)), loss_ratio, 
                   linewidth=2, color='green')
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss Ratio (vs Early Training)')
            ax.set_title('Learning Progress Indicator')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{model_name}_learning_curves.png')
        plt.close()
    
    def plot_eas_analysis(self, logger, model_name):
        """
        Analyse spécifique à EAS
        
        Args:
            logger: TrainingLogger avec les métriques
            model_name (str): Nom du modèle
        """
        if not logger.archive_sizes:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Archive Size
        ax = axes[0, 0]
        ax.plot(logger.archive_sizes, linewidth=2, color='purple')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Archive Size')
        ax.set_title('EAS Archive Growth')
        ax.grid(True, alpha=0.3)
        
        # 2. Q-Filter Rate
        ax = axes[0, 1]
        if logger.q_filter_rates:
            rates = np.array(logger.q_filter_rates) * 100
            ax.plot(rates, linewidth=2, color='green')
            smoothed = self.smooth_curve(rates, window=50)
            ax.plot(smoothed, linewidth=2, color='darkgreen', 
                   label='Smoothed', linestyle='--')
            ax.axhline(y=50, color='red', linestyle='--', alpha=0.5,
                      label='50% threshold')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Acceptance Rate (%)')
            ax.set_title('Q-Filter Acceptance Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Q-Values Comparison
        ax = axes[0, 2]
        if logger.q_values_normal and logger.q_values_evolved:
            steps = np.arange(len(logger.q_values_normal))
            ax.plot(steps, logger.q_values_normal, 
                   linewidth=2, label='Normal Actions', alpha=0.7)
            ax.plot(steps, logger.q_values_evolved, 
                   linewidth=2, label='Evolved Actions', alpha=0.7)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Average Q-Value')
            ax.set_title('Q-Values: Normal vs Evolved')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. PSO Improvement
        ax = axes[1, 0]
        if logger.pso_improvements:
            improvements = np.array(logger.pso_improvements)
            ax.plot(improvements, linewidth=1, alpha=0.5, color='red')
            smoothed = self.smooth_curve(improvements, window=50)
            ax.plot(smoothed, linewidth=2, color='darkred', label='Smoothed')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Q-Value Improvement')
            ax.set_title('PSO Action Improvement')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 5. Q-Value Improvement Distribution
        ax = axes[1, 1]
        if logger.pso_improvements:
            improvements = np.array(logger.pso_improvements)
            positive = improvements[improvements > 0]
            negative = improvements[improvements < 0]
            
            ax.hist(positive, bins=30, alpha=0.6, color='green', 
                   label=f'Positive ({len(positive)}/{len(improvements)})')
            ax.hist(negative, bins=30, alpha=0.6, color='red',
                   label=f'Negative ({len(negative)}/{len(improvements)})')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Q-Value Improvement')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of PSO Improvements')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 6. Evolved vs Normal Q-Values Scatter
        ax = axes[1, 2]
        if logger.q_values_normal and logger.q_values_evolved:
            normal = np.array(logger.q_values_normal)
            evolved = np.array(logger.q_values_evolved)
            
            ax.scatter(normal, evolved, alpha=0.5, s=20)
            
            # Ligne y=x
            min_val = min(normal.min(), evolved.min())
            max_val = max(normal.max(), evolved.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', linewidth=2, label='y=x')
            
            ax.set_xlabel('Normal Q-Values')
            ax.set_ylabel('Evolved Q-Values')
            ax.set_title('Q-Value Correlation')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{model_name}_eas_analysis.png')
        plt.close()
    
    def plot_exploration_analysis(self, logger, model_name):
        """
        Analyse de l'exploration
        
        Args:
            logger: TrainingLogger avec les métriques
            model_name (str): Nom du modèle
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Action Magnitudes
        ax = axes[0]
        if logger.action_magnitudes:
            steps = np.arange(len(logger.action_magnitudes))
            magnitudes = np.array(logger.action_magnitudes)
            
            ax.plot(steps, magnitudes, alpha=0.2, color='blue')
            smoothed = self.smooth_curve(magnitudes, window=1000)
            ax.plot(steps, smoothed, linewidth=2, color='darkblue',
                   label='Moving Average')
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Action Magnitude (L2 norm)')
            ax.set_title('Action Exploration Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Distribution des magnitudes par phase
        ax = axes[1]
        if len(logger.action_magnitudes) > 1000:
            n = len(logger.action_magnitudes)
            phases = [
                logger.action_magnitudes[:n//4],
                logger.action_magnitudes[n//4:n//2],
                logger.action_magnitudes[n//2:3*n//4],
                logger.action_magnitudes[3*n//4:]
            ]
            labels = ['Early (0-25%)', 'Mid-Early (25-50%)', 
                     'Mid-Late (50-75%)', 'Late (75-100%)']
            
            bp = ax.boxplot(phases, labels=labels, patch_artist=True)
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Action Magnitude')
            ax.set_title('Exploration by Training Phase')
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=15)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{model_name}_exploration_analysis.png')
        plt.close()
    
    def plot_comparison(self, logger_td3, logger_eas, name_td3, name_eas):
        """
        Comparaison TD3 vs EAS-TD3
        
        Args:
            logger_td3: Logger du modèle TD3
            logger_eas: Logger du modèle EAS-TD3
            name_td3 (str): Nom du modèle TD3
            name_eas (str): Nom du modèle EAS-TD3
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Récompenses d'évaluation
        ax = axes[0, 0]
        
        # TD3
        steps_td3 = np.array(logger_td3.eval_timesteps)
        rewards_td3 = np.array(logger_td3.eval_rewards)
        stds_td3 = np.array(logger_td3.eval_stds)
        
        ax.plot(steps_td3, rewards_td3, linewidth=2.5, 
                label='TD3', marker='o', markersize=4)
        ax.fill_between(steps_td3, 
                        rewards_td3 - stds_td3,
                        rewards_td3 + stds_td3,
                        alpha=0.3)
        
        # EAS-TD3
        steps_eas = np.array(logger_eas.eval_timesteps)
        rewards_eas = np.array(logger_eas.eval_rewards)
        stds_eas = np.array(logger_eas.eval_stds)
        
        ax.plot(steps_eas, rewards_eas, linewidth=2.5,
                label='EAS-TD3', marker='s', markersize=4)
        ax.fill_between(steps_eas,
                        rewards_eas - stds_eas,
                        rewards_eas + stds_eas,
                        alpha=0.3)
        
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Average Reward')
        ax.set_title('Evaluation Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Sample Efficiency (par épisode)
        ax = axes[0, 1]
        
        # Moyennes mobiles des récompenses par épisode
        episodes_td3 = np.arange(len(logger_td3.episode_rewards))
        ma_td3 = self.smooth_curve(logger_td3.episode_rewards, window=50)
        
        episodes_eas = np.arange(len(logger_eas.episode_rewards))
        ma_eas = self.smooth_curve(logger_eas.episode_rewards, window=50)
        
        ax.plot(episodes_td3, ma_td3, linewidth=2, label='TD3')
        ax.plot(episodes_eas, ma_eas, linewidth=2, label='EAS-TD3')
        
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Reward (MA-50)')
        ax.set_title('Sample Efficiency (Episode-wise)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Stabilité comparée
        ax = axes[1, 0]
        ax.plot(steps_td3, stds_td3, linewidth=2, 
                label='TD3', marker='o', markersize=4)
        ax.plot(steps_eas, stds_eas, linewidth=2,
                label='EAS-TD3', marker='s', markersize=4)
        
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Performance Stability Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Temps de convergence
        ax = axes[1, 1]
        
        # Calculer le temps pour atteindre différents seuils
        thresholds = np.linspace(min(rewards_td3.min(), rewards_eas.min()),
                                max(rewards_td3.max(), rewards_eas.max()),
                                10)
        
        convergence_td3 = []
        convergence_eas = []
        
        for threshold in thresholds:
            # TD3
            idx_td3 = np.where(rewards_td3 >= threshold)[0]
            conv_td3 = steps_td3[idx_td3[0]] if len(idx_td3) > 0 else steps_td3[-1]
            convergence_td3.append(conv_td3)
            
            # EAS-TD3
            idx_eas = np.where(rewards_eas >= threshold)[0]
            conv_eas = steps_eas[idx_eas[0]] if len(idx_eas) > 0 else steps_eas[-1]
            convergence_eas.append(conv_eas)
        
        ax.plot(thresholds, convergence_td3, linewidth=2, 
                label='TD3', marker='o')
        ax.plot(thresholds, convergence_eas, linewidth=2,
                label='EAS-TD3', marker='s')
        
        ax.set_xlabel('Performance Threshold (Reward)')
        ax.set_ylabel('Timesteps to Converge')
        ax.set_title('Convergence Speed')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/comparison_td3_vs_eas.png')
        plt.close()
        
        # Statistiques de comparaison
        self._print_comparison_stats(logger_td3, logger_eas, name_td3, name_eas)
    
    def _print_comparison_stats(self, logger_td3, logger_eas, name_td3, name_eas):
        """Affiche les statistiques de comparaison"""
        print("\n" + "="*70)
        print("COMPARISON STATISTICS: TD3 vs EAS-TD3")
        print("="*70)
        
        # Performance finale
        final_td3 = logger_td3.eval_rewards[-1]
        final_eas = logger_eas.eval_rewards[-1]
        improvement = ((final_eas - final_td3) / abs(final_td3)) * 100
        
        print(f"\nFinal Performance:")
        print(f"  TD3:     {final_td3:.2f}")
        print(f"  EAS-TD3: {final_eas:.2f}")
        print(f"  Improvement: {improvement:+.2f}%")
        
        # Stabilité
        stability_td3 = np.mean(logger_td3.eval_stds)
        stability_eas = np.mean(logger_eas.eval_stds)
        
        print(f"\nAverage Stability (Lower is Better):")
        print(f"  TD3:     {stability_td3:.2f}")
        print(f"  EAS-TD3: {stability_eas:.2f}")
        
        # Efficacité d'échantillonnage
        episodes_td3 = len(logger_td3.episode_rewards)
        episodes_eas = len(logger_eas.episode_rewards)
        
        print(f"\nTotal Episodes:")
        print(f"  TD3:     {episodes_td3}")
        print(f"  EAS-TD3: {episodes_eas}")
        
        # Spécifique EAS
        if logger_eas.q_filter_rates:
            avg_filter_rate = np.mean(logger_eas.q_filter_rates) * 100
            avg_improvement = np.mean(logger_eas.pso_improvements)
            
            print(f"\nEAS Statistics:")
            print(f"  Avg Q-Filter Rate: {avg_filter_rate:.2f}%")
            print(f"  Avg PSO Improvement: {avg_improvement:.4f}")
            print(f"  Final Archive Size: {logger_eas.archive_sizes[-1]}")
        
        print("="*70 + "\n")
    
    def plot_action_distribution_evolution(self, action_history, model_name, 
                                           num_intervals=5, num_samples=1000):
        """
        Plot l'évolution de la distribution des actions au fil du temps
        
        Args:
            action_history: Liste de tuples (timestep, action_normal, action_evolved)
            model_name (str): Nom du modèle
            num_intervals (int): Nombre d'intervalles de temps
            num_samples (int): Nombre d'échantillons par intervalle
        """
        if not action_history or len(action_history) < num_samples * num_intervals:
            print("  ⚠ Not enough action data for distribution plot")
            return
        
        # Créer les intervalles
        total_steps = len(action_history)
        interval_size = total_steps // num_intervals
        
        fig, axes = plt.subplots(2, num_intervals, figsize=(4*num_intervals, 8))
        
        for i in range(num_intervals):
            start_idx = i * interval_size
            end_idx = min((i + 1) * interval_size, total_steps)
            
            # Échantillonner aléatoirement dans cet intervalle
            interval_data = action_history[start_idx:end_idx]
            if len(interval_data) > num_samples:
                interval_data = np.random.choice(interval_data, num_samples, replace=False)
            
            # Extraire les actions (première et deuxième dimension)
            normal_actions = np.array([x[1][:2] for x in interval_data])
            evolved_actions = np.array([x[2][:2] for x in interval_data])
            
            # Plot actions normales (rouge)
            ax = axes[0, i]
            try:
                from scipy.stats import gaussian_kde
                xy = np.vstack([normal_actions[:, 0], normal_actions[:, 1]])
                z = gaussian_kde(xy)(xy)
                ax.scatter(normal_actions[:, 0], normal_actions[:, 1], 
                          c=z, s=10, cmap='Reds', alpha=0.6)
                ax.contour(normal_actions[:, 0].reshape(-1, 1), 
                          normal_actions[:, 1].reshape(-1, 1),
                          z.reshape(-1, 1), colors='red', alpha=0.3)
            except:
                ax.scatter(normal_actions[:, 0], normal_actions[:, 1], 
                          c='red', s=10, alpha=0.3)
            
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_xlabel('Action Dimension 0')
            if i == 0:
                ax.set_ylabel('Action Dimension 1')
            ax.set_title(f'step: {start_idx}−{end_idx}')
            ax.grid(True, alpha=0.3)
            
            # Plot actions évoluées (bleu)
            ax = axes[1, i]
            try:
                xy = np.vstack([evolved_actions[:, 0], evolved_actions[:, 1]])
                z = gaussian_kde(xy)(xy)
                ax.scatter(evolved_actions[:, 0], evolved_actions[:, 1], 
                          c=z, s=10, cmap='Blues', alpha=0.6)
                ax.contour(evolved_actions[:, 0].reshape(-1, 1), 
                          evolved_actions[:, 1].reshape(-1, 1),
                          z.reshape(-1, 1), colors='blue', alpha=0.3)
            except:
                ax.scatter(evolved_actions[:, 0], evolved_actions[:, 1], 
                          c='blue', s=10, alpha=0.3)
            
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])
            ax.set_xlabel('Action Dimension 0')
            if i == 0:
                ax.set_ylabel('Action Dimension 1')
            ax.set_title(f'step: {start_idx}−{end_idx}')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Action Distribution Evolution - {model_name}\n' +
                    'Red: Current Policy Actions | Blue: Evolved Actions',
                    fontsize=14, y=0.995)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{model_name}_action_distribution_evolution.png',
                   bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self, logger, model_name, use_eas=False, action_history=None):
        """
        Génère tous les graphiques pour un modèle
        
        Args:
            logger: TrainingLogger avec les métriques
            model_name (str): Nom du modèle
            use_eas (bool): Si EAS est utilisé
            action_history: Historique des actions (optionnel)
        """
        print(f"\nGenerating plots for {model_name}...")
        
        self.plot_performance_analysis(logger, model_name)
        print(f"  ✓ Performance analysis")
        
        self.plot_learning_curves(logger, model_name)
        print(f"  ✓ Learning curves")
        
        self.plot_exploration_analysis(logger, model_name)
        print(f"  ✓ Exploration analysis")
        
        if use_eas:
            self.plot_eas_analysis(logger, model_name)
            print(f"  ✓ EAS-specific analysis")
            
            if action_history:
                self.plot_action_distribution_evolution(action_history, model_name)
                print(f"  ✓ Action distribution evolution")
        
        print(f"  All plots saved to {self.save_dir}/")


def create_visualizations(logger_td3, logger_eas, name_td3, name_eas, 
                         action_history_eas=None, save_dir="results/figures"):
    """
    Crée toutes les visualisations pour TD3 et EAS-TD3
    
    Args:
        logger_td3: Logger du modèle TD3
        logger_eas: Logger du modèle EAS-TD3
        name_td3 (str): Nom du modèle TD3
        name_eas (str): Nom du modèle EAS-TD3
        action_history_eas: Historique des actions pour EAS
        save_dir (str): Dossier de sauvegarde
    """
    plotter = EnhancedPlotter(save_dir=save_dir)
    
    # Plots individuels
    plotter.generate_all_plots(logger_td3, name_td3, use_eas=False)
    plotter.generate_all_plots(logger_eas, name_eas, use_eas=True, 
                              action_history=action_history_eas)
    
    # Comparaison
    print("\nGenerating comparison plots...")
    plotter.plot_comparison(logger_td3, logger_eas, name_td3, name_eas)
    print("  ✓ Comparison analysis")
    
    print(f"\n✓ All visualizations complete!")
