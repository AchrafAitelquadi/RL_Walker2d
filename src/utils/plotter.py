"""
Visualisation avancée des résultats d'entraînement
Style publication scientifique
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from scipy import stats

# Configuration style publication scientifique
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid", {
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'axes.edgecolor': '.15',
    'axes.linewidth': 1.25
})

# Configuration matplotlib pour qualité publication
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Palette de couleurs professionnelle
COLORS = {
    'primary': '#2E86AB',      # Bleu professionnel
    'secondary': '#A23B72',    # Violet
    'success': '#06A77D',      # Vert
    'warning': '#F77F00',      # Orange
    'danger': '#D62828',       # Rouge
    'info': '#118AB2',         # Bleu clair
    'dark': '#2B2D42',         # Gris foncé
    'light': '#EDF2F4',        # Gris clair
}


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
    
    def compute_confidence_interval(self, data, window=100, confidence=0.95):
        """
        Calcule l'intervalle de confiance autour d'une courbe
        
        Args:
            data: Données
            window (int): Taille de la fenêtre pour le calcul
            confidence (float): Niveau de confiance (0.95 = 95%)
            
        Returns:
            tuple: (mean, lower_bound, upper_bound)
        """
        if len(data) < window:
            return data, data, data
        
        n = len(data)
        mean = np.zeros(n)
        std = np.zeros(n)
        
        for i in range(n):
            start = max(0, i - window // 2)
            end = min(n, i + window // 2)
            window_data = data[start:end]
            mean[i] = np.mean(window_data)
            std[i] = np.std(window_data)
        
        # Calcul de l'intervalle de confiance
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * std / np.sqrt(window)
        
        return mean, mean - margin, mean + margin
    
    def plot_performance_analysis(self, logger, model_name):
        """
        Analyse complète des performances avec style publication
        
        Args:
            logger: TrainingLogger avec les métriques
            model_name (str): Nom du modèle
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Récompenses par épisode avec intervalle de confiance
        ax = fig.add_subplot(gs[0, :2])
        episodes = np.arange(len(logger.episode_rewards))
        rewards = np.array(logger.episode_rewards)
        
        # Calculer moyenne et intervalle de confiance
        mean, lower, upper = self.compute_confidence_interval(rewards, window=100)
        
        # Plot avec zone de confiance
        ax.plot(episodes, rewards, alpha=0.2, color=COLORS['primary'], 
                linewidth=0.8, label='Raw Episode Reward')
        ax.plot(episodes, mean, color=COLORS['primary'], linewidth=2.5, 
                label='Moving Average', zorder=10)
        ax.fill_between(episodes, lower, upper, 
                        alpha=0.25, color=COLORS['primary'], 
                        label='95% Confidence Interval')
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Cumulative Reward', fontweight='bold')
        ax.set_title('Training Progress: Episode Rewards', pad=15)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 2. Évaluations avec bandes d'erreur
        ax = fig.add_subplot(gs[0, 2])
        timesteps = np.array(logger.eval_timesteps)
        eval_rewards = np.array(logger.eval_rewards)
        eval_stds = np.array(logger.eval_stds)
        
        ax.plot(timesteps, eval_rewards, color=COLORS['success'], 
                linewidth=2.5, marker='o', markersize=5, 
                markeredgecolor='white', markeredgewidth=1.5)
        ax.fill_between(timesteps, 
                        eval_rewards - eval_stds,
                        eval_rewards + eval_stds,
                        alpha=0.3, color=COLORS['success'], label='±1σ')
        ax.fill_between(timesteps, 
                        eval_rewards - 2*eval_stds,
                        eval_rewards + 2*eval_stds,
                        alpha=0.15, color=COLORS['success'], label='±2σ')
        
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel('Eval Reward', fontweight='bold')
        ax.set_title('Evaluation Performance', pad=15)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # 3. Courbe de convergence (smoothed avec gradient)
        ax = fig.add_subplot(gs[1, 0])
        if len(rewards) > 100:
            smoothed = self.smooth_curve(rewards, window=50)
            episodes_smooth = episodes
            
            # Gradient de couleur basé sur la valeur
            points = np.array([episodes_smooth, smoothed]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            norm = plt.Normalize(smoothed.min(), smoothed.max())
            from matplotlib.collections import LineCollection
            lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=3)
            lc.set_array(smoothed)
            line = ax.add_collection(lc)
            
            ax.set_xlim(episodes.min(), episodes.max())
            ax.set_ylim(smoothed.min() * 0.95, smoothed.max() * 1.05)
            
            cbar = fig.colorbar(line, ax=ax)
            cbar.set_label('Reward Value', fontweight='bold')
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Reward (MA-50)', fontweight='bold')
        ax.set_title('Convergence Curve', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 4. Stabilité temporelle
        ax = fig.add_subplot(gs[1, 1])
        if len(timesteps) > 0:
            ax.plot(timesteps, eval_stds, color=COLORS['warning'], 
                   linewidth=2.5, marker='s', markersize=5,
                   markeredgecolor='white', markeredgewidth=1.5)
            
            # Tendance linéaire
            z = np.polyfit(timesteps, eval_stds, 1)
            p = np.poly1d(z)
            ax.plot(timesteps, p(timesteps), "--", color=COLORS['dark'], 
                   alpha=0.7, linewidth=2, label=f'Trend (slope={z[0]:.2e})')
        
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel('Std Deviation', fontweight='bold')
        ax.set_title('Performance Stability', pad=15)
        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # 5. Distribution par quartile (violin plot)
        ax = fig.add_subplot(gs[1, 2])
        if len(rewards) > 100:
            n = len(rewards)
            quartiles = [
                rewards[:n//4],
                rewards[n//4:n//2],
                rewards[n//2:3*n//4],
                rewards[3*n//4:]
            ]
            labels = ['Q1\n(Early)', 'Q2', 'Q3', 'Q4\n(Late)']
            
            parts = ax.violinplot(quartiles, showmeans=True, showmedians=True)
            
            # Colorer les violons
            colors_vio = [COLORS['primary'], COLORS['info'], 
                         COLORS['success'], COLORS['warning']]
            for pc, color in zip(parts['bodies'], colors_vio):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            ax.set_xticks(np.arange(1, 5))
            ax.set_xticklabels(labels)
        
        ax.set_ylabel('Reward Distribution', fontweight='bold')
        ax.set_title('Reward by Training Phase', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 6. Taux d'amélioration (dérivée de la récompense)
        ax = fig.add_subplot(gs[2, 0])
        if len(eval_rewards) > 2:
            improvement_rate = np.diff(eval_rewards)
            timesteps_diff = timesteps[1:]
            
            colors_bar = [COLORS['success'] if x > 0 else COLORS['danger'] 
                         for x in improvement_rate]
            ax.bar(timesteps_diff, improvement_rate, width=timesteps[1]-timesteps[0],
                  color=colors_bar, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel('Reward Improvement', fontweight='bold')
        ax.set_title('Learning Rate (Δ Reward)', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # 7. Histogramme cumulatif des récompenses
        ax = fig.add_subplot(gs[2, 1])
        if len(rewards) > 0:
            counts, bins = np.histogram(rewards, bins=50)
            cdf = np.cumsum(counts) / np.sum(counts) * 100
            
            ax.hist(rewards, bins=50, alpha=0.6, color=COLORS['primary'],
                   edgecolor='black', linewidth=0.5, label='Frequency')
            ax2 = ax.twinx()
            ax2.plot(bins[1:], cdf, color=COLORS['danger'], linewidth=2.5,
                    marker='o', markersize=4, label='Cumulative %')
            ax2.set_ylabel('Cumulative Percentage', fontweight='bold', 
                          color=COLORS['danger'])
            ax2.tick_params(axis='y', labelcolor=COLORS['danger'])
            ax2.set_ylim([0, 105])
        
        ax.set_xlabel('Reward', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Reward Distribution', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 8. Percentiles temporels
        ax = fig.add_subplot(gs[2, 2])
        if len(rewards) > 100:
            window = 100
            p10, p50, p90 = [], [], []
            timesteps_window = []
            
            for i in range(window, len(rewards), window):
                window_data = rewards[i-window:i]
                p10.append(np.percentile(window_data, 10))
                p50.append(np.percentile(window_data, 50))
                p90.append(np.percentile(window_data, 90))
                timesteps_window.append(i)
            
            timesteps_window = np.array(timesteps_window)
            p10, p50, p90 = np.array(p10), np.array(p50), np.array(p90)
            
            ax.fill_between(timesteps_window, p10, p90, 
                           alpha=0.3, color=COLORS['info'], label='10th-90th percentile')
            ax.plot(timesteps_window, p50, color=COLORS['dark'], 
                   linewidth=2.5, label='Median (50th)', marker='D', markersize=4)
            ax.plot(timesteps_window, p10, color=COLORS['info'], 
                   linewidth=1.5, linestyle='--', alpha=0.7)
            ax.plot(timesteps_window, p90, color=COLORS['info'], 
                   linewidth=1.5, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Episode', fontweight='bold')
        ax.set_ylabel('Reward Percentiles', fontweight='bold')
        ax.set_title('Performance Percentiles', pad=15)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'Performance Analysis - {model_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(f'{self.save_dir}/{model_name}_performance_analysis.png',
                   dpi=300, bbox_inches='tight')
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
        Comparaison avancée TD3 vs EAS-TD3 (style publication)
        
        Args:
            logger_td3: Logger du modèle TD3
            logger_eas: Logger du modèle EAS-TD3
            name_td3 (str): Nom du modèle TD3
            name_eas (str): Nom du modèle EAS-TD3
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # Données
        steps_td3 = np.array(logger_td3.eval_timesteps)
        rewards_td3 = np.array(logger_td3.eval_rewards)
        stds_td3 = np.array(logger_td3.eval_stds)
        
        steps_eas = np.array(logger_eas.eval_timesteps)
        rewards_eas = np.array(logger_eas.eval_rewards)
        stds_eas = np.array(logger_eas.eval_stds)
        
        # 1. Performance principale avec zones de confiance
        ax = fig.add_subplot(gs[0, :2])
        
        # TD3
        ax.plot(steps_td3, rewards_td3, linewidth=3, 
                color=COLORS['primary'], label='TD3', 
                marker='o', markersize=6, markeredgecolor='white', 
                markeredgewidth=1.5, zorder=5)
        ax.fill_between(steps_td3, 
                        rewards_td3 - stds_td3,
                        rewards_td3 + stds_td3,
                        alpha=0.25, color=COLORS['primary'], label='TD3 ±1σ')
        ax.fill_between(steps_td3, 
                        rewards_td3 - 2*stds_td3,
                        rewards_td3 + 2*stds_td3,
                        alpha=0.1, color=COLORS['primary'])
        
        # EAS-TD3
        ax.plot(steps_eas, rewards_eas, linewidth=3,
                color=COLORS['secondary'], label='EAS-TD3', 
                marker='s', markersize=6, markeredgecolor='white', 
                markeredgewidth=1.5, zorder=5)
        ax.fill_between(steps_eas,
                        rewards_eas - stds_eas,
                        rewards_eas + stds_eas,
                        alpha=0.25, color=COLORS['secondary'], label='EAS-TD3 ±1σ')
        ax.fill_between(steps_eas,
                        rewards_eas - 2*stds_eas,
                        rewards_eas + 2*stds_eas,
                        alpha=0.1, color=COLORS['secondary'])
        
        # Annotations
        if len(rewards_td3) > 0 and len(rewards_eas) > 0:
            final_td3 = rewards_td3[-1]
            final_eas = rewards_eas[-1]
            improvement = ((final_eas - final_td3) / abs(final_td3)) * 100
            
            ax.annotate(f'TD3 Final: {final_td3:.1f}', 
                       xy=(steps_td3[-1], final_td3),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc=COLORS['primary'], alpha=0.7),
                       color='white', fontweight='bold', fontsize=10)
            
            ax.annotate(f'EAS-TD3 Final: {final_eas:.1f}\n(+{improvement:.1f}%)', 
                       xy=(steps_eas[-1], final_eas),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc=COLORS['secondary'], alpha=0.7),
                       color='white', fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel('Evaluation Reward', fontweight='bold')
        ax.set_title('Performance Comparison: TD3 vs EAS-TD3', fontsize=14, fontweight='bold', pad=15)
        ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # 2. Amélioration relative (%)
        ax = fig.add_subplot(gs[0, 2])
        
        # Interpoler pour avoir les mêmes timesteps
        common_steps = np.intersect1d(steps_td3, steps_eas)
        if len(common_steps) > 0:
            idx_td3 = np.isin(steps_td3, common_steps)
            idx_eas = np.isin(steps_eas, common_steps)
            
            improvement_pct = ((rewards_eas[idx_eas] - rewards_td3[idx_td3]) / 
                              np.abs(rewards_td3[idx_td3])) * 100
            
            colors_bar = [COLORS['success'] if x > 0 else COLORS['danger'] 
                         for x in improvement_pct]
            ax.bar(common_steps, improvement_pct, 
                  width=(common_steps[1] - common_steps[0]) * 0.8,
                  color=colors_bar, edgecolor='black', linewidth=0.5, alpha=0.8)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
            
            # Ligne de tendance
            z = np.polyfit(common_steps, improvement_pct, 2)
            p = np.poly1d(z)
            ax.plot(common_steps, p(common_steps), '--', 
                   color=COLORS['dark'], linewidth=2.5, alpha=0.7, 
                   label='Polynomial Trend')
        
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel('Improvement (%)', fontweight='bold')
        ax.set_title('Relative Improvement', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', framealpha=0.95, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # 3. Sample Efficiency (Episode-wise)
        ax = fig.add_subplot(gs[1, 0])
        
        episodes_td3 = np.arange(len(logger_td3.episode_rewards))
        episodes_eas = np.arange(len(logger_eas.episode_rewards))
        
        # Calculer moyennes avec IC
        mean_td3, lower_td3, upper_td3 = self.compute_confidence_interval(
            np.array(logger_td3.episode_rewards), window=100)
        mean_eas, lower_eas, upper_eas = self.compute_confidence_interval(
            np.array(logger_eas.episode_rewards), window=100)
        
        ax.plot(episodes_td3, mean_td3, linewidth=2.5, 
               color=COLORS['primary'], label='TD3', alpha=0.9)
        ax.fill_between(episodes_td3, lower_td3, upper_td3,
                       alpha=0.2, color=COLORS['primary'])
        
        ax.plot(episodes_eas, mean_eas, linewidth=2.5,
               color=COLORS['secondary'], label='EAS-TD3', alpha=0.9)
        ax.fill_between(episodes_eas, lower_eas, upper_eas,
                       alpha=0.2, color=COLORS['secondary'])
        
        ax.set_xlabel('Episodes', fontweight='bold')
        ax.set_ylabel('Reward (95% CI)', fontweight='bold')
        ax.set_title('Sample Efficiency', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 4. Stabilité comparée
        ax = fig.add_subplot(gs[1, 1])
        
        ax.plot(steps_td3, stds_td3, linewidth=2.5, 
               color=COLORS['primary'], marker='o', markersize=5,
               markeredgecolor='white', markeredgewidth=1, 
               label='TD3')
        ax.plot(steps_eas, stds_eas, linewidth=2.5,
               color=COLORS['secondary'], marker='s', markersize=5,
               markeredgecolor='white', markeredgewidth=1,
               label='EAS-TD3')
        
        # Moyennes
        mean_std_td3 = np.mean(stds_td3)
        mean_std_eas = np.mean(stds_eas)
        ax.axhline(y=mean_std_td3, color=COLORS['primary'], 
                  linestyle='--', alpha=0.5, linewidth=2,
                  label=f'TD3 Avg: {mean_std_td3:.1f}')
        ax.axhline(y=mean_std_eas, color=COLORS['secondary'], 
                  linestyle='--', alpha=0.5, linewidth=2,
                  label=f'EAS Avg: {mean_std_eas:.1f}')
        
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel('Standard Deviation', fontweight='bold')
        ax.set_title('Performance Stability', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', framealpha=0.95, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # 5. Convergence Speed
        ax = fig.add_subplot(gs[1, 2])
        
        # Calculer temps pour atteindre différents seuils
        max_reward = max(rewards_td3.max(), rewards_eas.max())
        min_reward = min(rewards_td3.min(), rewards_eas.min())
        thresholds = np.linspace(min_reward, max_reward * 0.95, 8)
        
        convergence_td3 = []
        convergence_eas = []
        achieved_thresholds = []
        
        for threshold in thresholds:
            idx_td3 = np.where(rewards_td3 >= threshold)[0]
            idx_eas = np.where(rewards_eas >= threshold)[0]
            
            if len(idx_td3) > 0 and len(idx_eas) > 0:
                convergence_td3.append(steps_td3[idx_td3[0]])
                convergence_eas.append(steps_eas[idx_eas[0]])
                achieved_thresholds.append(threshold)
        
        if len(achieved_thresholds) > 0:
            achieved_thresholds = np.array(achieved_thresholds)
            convergence_td3 = np.array(convergence_td3)
            convergence_eas = np.array(convergence_eas)
            
            ax.plot(achieved_thresholds, convergence_td3, linewidth=2.5, 
                   color=COLORS['primary'], marker='o', markersize=6,
                   markeredgecolor='white', markeredgewidth=1.5, label='TD3')
            ax.plot(achieved_thresholds, convergence_eas, linewidth=2.5,
                   color=COLORS['secondary'], marker='s', markersize=6,
                   markeredgecolor='white', markeredgewidth=1.5, label='EAS-TD3')
            
            # Zone entre les courbes
            ax.fill_between(achieved_thresholds, convergence_td3, convergence_eas,
                           where=(convergence_eas < convergence_td3),
                           alpha=0.3, color=COLORS['success'], 
                           label='EAS-TD3 Faster')
            ax.fill_between(achieved_thresholds, convergence_td3, convergence_eas,
                           where=(convergence_eas >= convergence_td3),
                           alpha=0.3, color=COLORS['danger'],
                           label='TD3 Faster')
        
        ax.set_xlabel('Performance Threshold (Reward)', fontweight='bold')
        ax.set_ylabel('Timesteps to Converge', fontweight='bold')
        ax.set_title('Convergence Speed', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        # 6. Distribution finale des récompenses
        ax = fig.add_subplot(gs[2, 0])
        
        if len(logger_td3.episode_rewards) > 50 and len(logger_eas.episode_rewards) > 50:
            # Derniers 100 épisodes
            final_td3 = logger_td3.episode_rewards[-100:]
            final_eas = logger_eas.episode_rewards[-100:]
            
            parts_td3 = ax.violinplot([final_td3], positions=[1], 
                                      showmeans=True, showmedians=True, widths=0.7)
            parts_eas = ax.violinplot([final_eas], positions=[2], 
                                      showmeans=True, showmedians=True, widths=0.7)
            
            # Colorer
            for pc in parts_td3['bodies']:
                pc.set_facecolor(COLORS['primary'])
                pc.set_alpha(0.7)
            for pc in parts_eas['bodies']:
                pc.set_facecolor(COLORS['secondary'])
                pc.set_alpha(0.7)
            
            ax.set_xticks([1, 2])
            ax.set_xticklabels(['TD3', 'EAS-TD3'])
        
        ax.set_ylabel('Reward Distribution', fontweight='bold')
        ax.set_title('Final Performance\n(Last 100 Episodes)', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # 7. Learning Rate (dérivée)
        ax = fig.add_subplot(gs[2, 1])
        
        if len(rewards_td3) > 2 and len(rewards_eas) > 2:
            # Calculer dérivées (taux d'apprentissage)
            lr_td3 = np.diff(rewards_td3) / np.diff(steps_td3)
            lr_eas = np.diff(rewards_eas) / np.diff(steps_eas)
            
            ax.plot(steps_td3[1:], lr_td3, linewidth=2, 
                   color=COLORS['primary'], alpha=0.7, label='TD3')
            ax.plot(steps_eas[1:], lr_eas, linewidth=2,
                   color=COLORS['secondary'], alpha=0.7, label='EAS-TD3')
            
            # Smooth
            if len(lr_td3) > 5:
                lr_td3_smooth = self.smooth_curve(lr_td3, window=min(5, len(lr_td3)//2))
                lr_eas_smooth = self.smooth_curve(lr_eas, window=min(5, len(lr_eas)//2))
                ax.plot(steps_td3[1:], lr_td3_smooth, linewidth=3, 
                       color=COLORS['primary'], linestyle='--')
                ax.plot(steps_eas[1:], lr_eas_smooth, linewidth=3,
                       color=COLORS['secondary'], linestyle='--')
            
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        ax.set_xlabel('Timesteps', fontweight='bold')
        ax.set_ylabel('Learning Rate (∂R/∂t)', fontweight='bold')
        ax.set_title('Instantaneous Learning Rate', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        
        # 8. Statistiques finales (texte)
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        
        # Calculer statistiques
        final_reward_td3 = rewards_td3[-1] if len(rewards_td3) > 0 else 0
        final_reward_eas = rewards_eas[-1] if len(rewards_eas) > 0 else 0
        improvement = ((final_reward_eas - final_reward_td3) / 
                      abs(final_reward_td3)) * 100 if final_reward_td3 != 0 else 0
        
        avg_std_td3 = np.mean(stds_td3) if len(stds_td3) > 0 else 0
        avg_std_eas = np.mean(stds_eas) if len(stds_eas) > 0 else 0
        stability_gain = ((avg_std_td3 - avg_std_eas) / avg_std_td3) * 100 if avg_std_td3 != 0 else 0
        
        total_episodes_td3 = len(logger_td3.episode_rewards)
        total_episodes_eas = len(logger_eas.episode_rewards)
        
        # Texte formaté
        stats_text = f"""
COMPARISON STATISTICS
{'='*35}

Final Performance:
  TD3:        {final_reward_td3:8.2f} ± {stds_td3[-1]:.2f}
  EAS-TD3:    {final_reward_eas:8.2f} ± {stds_eas[-1]:.2f}
  
  Improvement: {improvement:+7.2f}%

Stability (Avg Std Dev):
  TD3:        {avg_std_td3:8.2f}
  EAS-TD3:    {avg_std_eas:8.2f}
  
  Gain:       {stability_gain:+7.2f}%

Sample Efficiency:
  TD3:        {total_episodes_td3:5d} episodes
  EAS-TD3:    {total_episodes_eas:5d} episodes
        """
        
        # EAS stats
        if logger_eas.archive_sizes:
            avg_filter = np.mean(logger_eas.q_filter_rates) * 100 if logger_eas.q_filter_rates else 0
            final_archive = logger_eas.archive_sizes[-1]
            stats_text += f"""
EAS-Specific:
  Archive Size:     {final_archive:7,}
  Q-Filter Rate:    {avg_filter:7.1f}%
"""
        
        ax.text(0.1, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', 
               family='monospace',
               bbox=dict(boxstyle='round', facecolor=COLORS['light'], 
                        edgecolor=COLORS['dark'], linewidth=2, alpha=0.9))
        
        plt.suptitle('Comprehensive Comparison: TD3 vs EAS-TD3', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.savefig(f'{self.save_dir}/comparison_td3_vs_eas.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Afficher statistiques dans le terminal
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
            print("  [WARNING] Not enough action data for distribution plot")
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
        print(f"  [OK] Performance analysis")
        
        self.plot_learning_curves(logger, model_name)
        print(f"  [OK] Learning curves")
        
        self.plot_exploration_analysis(logger, model_name)
        print(f"  [OK] Exploration analysis")
        
        if use_eas:
            self.plot_eas_analysis(logger, model_name)
            print(f"  [OK] EAS-specific analysis")
            
            if action_history:
                self.plot_action_distribution_evolution(action_history, model_name)
                print(f"  [OK] Action distribution evolution")
        
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
    print("  [OK] Comparison analysis")
    
    print(f"\n[OK] All visualizations complete!")
