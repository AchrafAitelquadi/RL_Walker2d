"""
Script principal pour reproduire les expériences TD3 et EAS-TD3
Utilise les paramètres définis dans config.py

Pour modifier la configuration, éditez src/config.py ou utilisez les arguments CLI
"""

import os
import sys
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
import argparse
import time

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from algorithms import TD3, ReplayBuffer
from utils import TrainingLogger, EnhancedPlotter, create_visualizations
import config  # Importer la configuration


def format_time(seconds):
    """Formate les secondes en format lisible (HH:MM:SS)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    elif minutes > 0:
        return f"{minutes}m{secs:02d}s"
    else:
        return f"{secs}s"


def print_progress_bar(current, total, start_time, bar_length=40, prefix='Progress'):
    """
    Affiche une barre de progression avec ETA
    
    Args:
        current (int): Timestep actuel
        total (int): Total timesteps
        start_time (float): Temps de départ (time.time())
        bar_length (int): Longueur de la barre
        prefix (str): Préfixe à afficher
    """
    percent = float(current) / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Calculer ETA
    elapsed = time.time() - start_time
    if current > 0:
        eta = elapsed * (total - current) / current
        eta_str = format_time(eta)
    else:
        eta_str = "Calculating..."
    
    # Afficher
    print(f'\r{prefix}: |{bar}| {percent*100:.1f}% ({current:,}/{total:,}) | '
          f'Elapsed: {format_time(elapsed)} | ETA: {eta_str}', end='', flush=True)


def evaluate_policy(agent, env, seed, eval_episodes=None):
    """
    Évalue la politique actuelle sur plusieurs épisodes
    
    Args:
        agent: Agent TD3 à évaluer
        env: Environnement Gymnasium
        seed (int): Seed pour la reproductibilité
        eval_episodes (int): Nombre d'épisodes d'évaluation
        
    Returns:
        tuple: (récompense moyenne, écart-type)
    """
    if eval_episodes is None:
        eval_episodes = getattr(config, 'EVAL_EPISODES', 10)
    rewards = []
    for _ in range(eval_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)
    
    return np.mean(rewards), np.std(rewards)


def train_agent(env_name, use_eas, max_timesteps, eval_freq, seed,
                save_dir, log_dir, fig_dir):
    """
    Entraîne un agent TD3 ou EAS-TD3
    
    Args:
        env_name (str): Nom de l'environnement Gymnasium
        use_eas (bool): Utiliser EAS pour améliorer TD3
        max_timesteps (int): Nombre total de pas d'entraînement
        eval_freq (int): Fréquence d'évaluation (en timesteps)
        seed (int): Seed pour la reproductibilité
        save_dir (str): Dossier pour sauvegarder les modèles
        log_dir (str): Dossier pour les logs
        fig_dir (str): Dossier pour les figures
        
    Returns:
        tuple: (logger, model_name, action_history)
    """
    # Créer dossiers de sauvegarde
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{'EAS_' if use_eas else ''}TD3_{env_name}_{timestamp}"
    
    # Créer environnement
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    
    # Seeds pour reproductibilité
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    if config.VERBOSE:
        print(f"\n{'='*70}")
        print(f"Training {'EAS-' if use_eas else ''}TD3 on {env_name}")
        print(f"{'='*70}")
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Max action: {max_action}")
        print(f"Max timesteps: {max_timesteps:,}")
        print(f"Seed: {seed}")
        print(f"{'='*70}\n")
    
    # Agent et buffers
    agent = TD3(state_dim, action_dim, max_action, use_eas=use_eas)
    replay_buffer = ReplayBuffer()
    logger = TrainingLogger()
    
    state, _ = env.reset(seed=seed)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    
    pso_improvement_accumulator = []
    action_history = []
    
    # Timer pour la progression
    start_time = time.time()
    last_print_time = start_time
    print_interval = 1.0  # Mettre à jour la barre toutes les 1 seconde
    
    for t in range(max_timesteps):
        episode_timesteps += 1
        
        # Sélection d'action
        if t < config.START_TIMESTEPS:  # Phase d'exploration initiale
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
            action = action + np.random.normal(0, config.EXPL_NOISE * max_action, size=action_dim)
            action = action.clip(-max_action, max_action)
        
        # Log action magnitude
        action_magnitude = np.linalg.norm(action)
        logger.log_action(action_magnitude)
        
        # Exécuter l'action dans l'environnement
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Stocker dans le replay buffer
        replay_buffer.add(state, action, reward, next_state, float(done))
        
        # EAS: Évoluer l'action et stocker dans l'archive
        if use_eas and t >= config.START_TIMESTEPS:
            evo_action, improvement = agent.eas.evolve_action(state, action)
            agent.archive.add(state, evo_action)
            pso_improvement_accumulator.append(improvement)
            if t % 100 == 0:
                action_history.append((t, action.copy(), evo_action.copy()))
        
        state = next_state
        episode_reward += reward
        
        # Entraîner l'agent
        if t >= config.START_TIMESTEPS:
            critic_loss, actor_loss, q_filter_rate, q_normal, q_evolved = agent.train(
                replay_buffer,
                batch_size=config.BATCH_SIZE,
                discount=config.DISCOUNT,
                tau=config.TAU,
                policy_noise=config.POLICY_NOISE,
                noise_clip=config.NOISE_CLIP,
                policy_freq=config.POLICY_FREQ
            )
            
            if actor_loss is not None:
                logger.log_training(critic_loss, actor_loss, replay_buffer.size())
                
                if use_eas and agent.archive.size() > 0:
                    pso_improvement = np.mean(pso_improvement_accumulator) if pso_improvement_accumulator else 0
                    logger.log_eas(agent.archive.size(), q_filter_rate, q_normal, q_evolved, pso_improvement)
                    pso_improvement_accumulator = []
        
        if done:
            logger.log_episode(episode_reward)
            if config.VERBOSE:
                # Nettoyer la barre de progression avant le print
                print('\r' + ' ' * 120 + '\r', end='')
                print(f"Episode {episode_num+1} | Timestep: {t+1:,} | Reward: {episode_reward:.2f} | "
                      f"Avg(100): {np.mean(logger.episode_rewards[-100:]):.2f}")
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1
        
        # Afficher barre de progression périodiquement
        current_time = time.time()
        if current_time - last_print_time >= print_interval or t == max_timesteps - 1:
            if not config.VERBOSE:
                print_progress_bar(t + 1, max_timesteps, start_time, 
                                 prefix=f"{'EAS-' if use_eas else ''}TD3")
            last_print_time = current_time
        
        # Évaluation périodique
        if (t + 1) % eval_freq == 0:
            eval_reward, eval_std = evaluate_policy(agent, eval_env, seed)
            logger.log_eval(t + 1, eval_reward, eval_std)
            
            # Nettoyer la ligne et afficher l'évaluation
            print('\r' + ' ' * 120 + '\r', end='')
            print(f"\n{'='*70}")
            print(f"EVALUATION at {t+1:,}/{max_timesteps:,} timesteps ({(t+1)/max_timesteps*100:.1f}%)")
            print(f"{'='*70}")
            print(f"   Eval Reward: {eval_reward:.2f} ± {eval_std:.2f}")
            if logger.episode_rewards:
                print(f"   Train Reward (last 100 eps): {np.mean(logger.episode_rewards[-100:]):.2f}")
            if use_eas and logger.archive_sizes:
                print(f"   Archive Size: {logger.archive_sizes[-1]:,}")
                if logger.pso_improvements:
                    print(f"   PSO Improvement: {logger.pso_improvements[-1]:.4f}")
            elapsed = time.time() - start_time
            print(f"   Elapsed Time: {format_time(elapsed)}")
            print(f"{'='*70}\n")
        
        # Sauvegarder checkpoints si activé
        if config.SAVE_CHECKPOINTS and (t + 1) % config.CHECKPOINT_FREQ == 0:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_checkpoint_{t+1}.pt")
            agent.save(checkpoint_path)
            # Nettoyer la ligne avant le print
            print('\r' + ' ' * 120 + '\r', end='')
            print(f"Checkpoint saved at {t+1:,} timesteps")
    
    # Nettoyer la barre de progression finale
    print('\r' + ' ' * 120 + '\r', end='')
    
    # Sauvegarder le modèle final
    model_path = os.path.join(save_dir, f"{model_name}.pt")
    agent.save(model_path)
    
    # Fermer les environnements
    env.close()
    eval_env.close()
    
    # Statistiques finales
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"   Model: {model_path}")
    print(f"   Total Episodes: {episode_num}")
    print(f"   Total Time: {format_time(total_time)}")
    print(f"   Speed: {max_timesteps/total_time:.0f} timesteps/sec")
    if logger.eval_rewards:
        print(f"   Final Eval Reward: {logger.eval_rewards[-1]:.2f}")
    print(f"{'='*70}\n")
    
    return logger, model_name, action_history


def parse_args():
    """
    Parse les arguments de ligne de commande pour surcharger config.py
    """
    parser = argparse.ArgumentParser(
        description='Train TD3/EAS-TD3 agents on continuous control tasks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Sélection des algorithmes
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['td3', 'eas-td3'],
                       help='Algorithms to train (overrides config.py)')
    
    # Paramètres d'entraînement
    parser.add_argument('--timesteps', type=int, 
                       help='Total training timesteps')
    parser.add_argument('--seed', type=int, 
                       help='Random seed')
    parser.add_argument('--env', type=str, 
                       help='Gymnasium environment name')
    
    # Paramètres d'évaluation
    parser.add_argument('--eval-freq', type=int, 
                       help='Evaluation frequency (timesteps)')
    parser.add_argument('--eval-episodes', type=int, 
                       help='Number of evaluation episodes')
    
    # Options de sauvegarde
    parser.add_argument('--save-dir', type=str, 
                       help='Directory to save models')
    parser.add_argument('--no-checkpoints', action='store_true',
                       help='Disable checkpoint saving')
    
    # Options de visualisation
    parser.add_argument('--no-plots', action='store_true',
                       help='Disable plot generation')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output')
    
    return parser.parse_args()


def apply_cli_overrides(args):
    """
    Applique les arguments CLI pour surcharger config.py
    """
    if args.algorithms:
        config.TRAIN_TD3 = 'td3' in args.algorithms
        config.TRAIN_EAS_TD3 = 'eas-td3' in args.algorithms
    
    if args.timesteps:
        config.MAX_TIMESTEPS = args.timesteps
    
    if args.seed is not None:
        config.SEED = args.seed
    
    if args.env:
        config.ENV_NAME = args.env
    
    if args.eval_freq:
        config.EVAL_FREQ = args.eval_freq
    
    if args.eval_episodes:
        config.EVAL_EPISODES = args.eval_episodes
    
    if args.save_dir:
        config.SAVE_DIR = args.save_dir
    
    if args.no_checkpoints:
        config.SAVE_CHECKPOINTS = False
    
    if args.no_plots:
        config.PLOT_PERFORMANCE = False
        config.PLOT_COMPARISON = False
    
    if args.verbose:
        config.VERBOSE = True
    
    if args.quiet:
        config.VERBOSE = False


def main():
    """
    Point d'entrée principal pour reproduire les expériences
    
    Utilise la configuration définie dans config.py
    Les arguments CLI peuvent surcharger la configuration
    """
    # Parser les arguments CLI
    args = parse_args()
    
    # Appliquer les surcharges CLI
    apply_cli_overrides(args)
    
    # Valider et afficher la configuration finale
    config.validate_config()
    config.print_config()
    
    # Dictionnaire pour stocker les résultats de tous les algorithmes
    results = {
        'td3': {'logger': None, 'model_name': None, 'action_history': None},
        'eas-td3': {'logger': None, 'model_name': None, 'action_history': None}
    }
    
    # Liste des algorithmes à entraîner
    algorithms_to_train = []
    if config.TRAIN_TD3:
        algorithms_to_train.append('td3')
    if config.TRAIN_EAS_TD3:
        algorithms_to_train.append('eas-td3')
    
    if not algorithms_to_train:
        print("WARNING: No algorithms selected for training!")
        print("   Set TRAIN_TD3 or TRAIN_EAS_TD3 in config.py or use --algorithms CLI argument")
        return
    
    num_algorithms = len(algorithms_to_train)
    
    # ==================== ENTRAÎNEMENT ====================
    
    for idx, algo in enumerate(algorithms_to_train, 1):
        use_eas = (algo == 'eas-td3')
        algo_name = 'EAS-TD3' if use_eas else 'TD3'
        
        print("\n" + "▶"*35)
        if num_algorithms > 1:
            print(f"STEP {idx}/{num_algorithms}: Training {algo_name}")
        else:
            print(f"Training {algo_name}")
        print("▶"*35 + "\n")
        
        logger, model_name, action_history = train_agent(
            env_name=config.ENV_NAME,
            use_eas=use_eas,
            max_timesteps=config.MAX_TIMESTEPS,
            eval_freq=config.EVAL_FREQ,
            seed=config.SEED,
            save_dir=config.SAVE_DIR,
            log_dir=config.LOG_DIR,
            fig_dir=config.FIG_DIR
        )
        
        results[algo] = {
            'logger': logger,
            'model_name': model_name,
            'action_history': action_history
        }
        
        print(f"\n[OK] {algo_name} training complete!")
    
    # ==================== VISUALISATIONS ====================
    
    if config.PLOT_PERFORMANCE or config.PLOT_COMPARISON:
        print("\n" + "▶"*35)
        
        # Déterminer le type de visualisation
        if num_algorithms > 1 and config.PLOT_COMPARISON:
            # Plusieurs algorithmes: générer comparaison + plots individuels
            print("Generating comparison plots and individual plots")
            print("▶"*35 + "\n")
            
            create_visualizations(
                results['td3']['logger'],
                results['eas-td3']['logger'],
                results['td3']['model_name'],
                results['eas-td3']['model_name'],
                action_history_eas=results['eas-td3']['action_history'],
                save_dir=config.FIG_DIR
            )
            
        else:
            # Un seul algorithme ou comparaison désactivée: plots individuels uniquement
            print("Generating individual plots")
            print("▶"*35 + "\n")
            
            plotter = EnhancedPlotter(save_dir=config.FIG_DIR)
            
            for algo in algorithms_to_train:
                if results[algo]['logger'] is not None:
                    use_eas = (algo == 'eas-td3')
                    plotter.generate_all_plots(
                        results[algo]['logger'],
                        results[algo]['model_name'],
                        use_eas=use_eas,
                        action_history=results[algo]['action_history']
                    )
    
    # ==================== RÉSUMÉ FINAL ====================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    for algo in algorithms_to_train:
        algo_name = 'EAS-TD3' if algo == 'eas-td3' else 'TD3'
        model_name = results[algo]['model_name']
        logger = results[algo]['logger']
        
        print(f"\n{algo_name} Model:")
        print(f"   {config.SAVE_DIR}/{model_name}.pt")
        if logger and logger.eval_rewards:
            print(f"   Final reward: {logger.eval_rewards[-1]:.2f}")
    
    # Afficher amélioration si les deux algorithmes ont été entraînés
    if num_algorithms > 1 and all(results[a]['logger'] is not None for a in algorithms_to_train):
        if results['td3']['logger'].eval_rewards and results['eas-td3']['logger'].eval_rewards:
            td3_final = results['td3']['logger'].eval_rewards[-1]
            eas_final = results['eas-td3']['logger'].eval_rewards[-1]
            improvement = ((eas_final - td3_final) / abs(td3_final)) * 100
            print(f"\nEAS-TD3 vs TD3 Improvement: {improvement:+.2f}%")
    
    print(f"\nResults saved to: {config.FIG_DIR}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
