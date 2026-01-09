"""
Configuration centralisée pour les expériences TD3/EAS-TD3

Modifiez ce fichier pour configurer vos expériences
"""

# ======================== ALGORITHMES ========================
# Choisissez quels algorithmes entraîner

TRAIN_TD3 = False           # True: Entraîner TD3 standard
TRAIN_EAS_TD3 = True        # True: Entraîner EAS-TD3

# Si les deux sont True: entraînement séquentiel + plots de comparaison
# Si un seul est True: entraînement simple de cet algorithme
# =============================================================


# ======================== ENVIRONNEMENT ======================
ENV_NAME = "Walker2d-v4"    # Nom de l'environnement Gymnasium
# =============================================================


# ======================== ENTRAÎNEMENT =======================
MAX_TIMESTEPS = 1000000     # Timesteps par algorithme
                            # TD3: ~2-4h sur CPU
                            # Les deux: ~4-8h sur CPU

START_TIMESTEPS = 25000     # Phase d'exploration aléatoire initiale
EVAL_FREQ = 5000            # Fréquence d'évaluation (en timesteps)
BATCH_SIZE = 100            # Taille du batch pour l'entraînement
SEED = 0                    # Seed pour reproductibilité
# =============================================================


# ======================== HYPERPARAMÈTRES ====================
# TD3
DISCOUNT = 0.99             # Facteur de discount (gamma)
TAU = 0.005                 # Coefficient de soft update
POLICY_NOISE = 0.2          # Bruit pour target policy smoothing
NOISE_CLIP = 0.5            # Clipping du bruit
POLICY_FREQ = 2             # Fréquence de mise à jour de l'actor
LEARNING_RATE = 3e-4        # Learning rate (Adam)
EXPL_NOISE = 0.1            # Bruit d'exploration (std)

# EAS (Particle Swarm Optimization)
PSO_POP_SIZE = 10           # Taille de la population PSO
PSO_ITERATIONS = 10         # Nombre d'itérations PSO
PSO_OMEGA = 1.2             # Coefficient d'inertie
PSO_C1 = 1.5                # Coefficient cognitif (personal best)
PSO_C2 = 1.5                # Coefficient social (global best)
PSO_VMAX = 0.1              # Vélocité maximale (× max_action)
# =============================================================


# ======================== SAUVEGARDE =========================
SAVE_DIR = "models"         # Dossier pour les modèles
LOG_DIR = "results/logs"    # Dossier pour les logs
FIG_DIR = "results/figures" # Dossier pour les figures
# =============================================================


# ======================== VISUALISATION ======================
PLOT_PERFORMANCE = True     # Graphiques de performance
PLOT_LEARNING = True        # Courbes d'apprentissage
PLOT_EAS_ANALYSIS = True    # Analyse EAS (si applicable)
PLOT_EXPLORATION = True     # Analyse de l'exploration
PLOT_COMPARISON = True      # Comparaison TD3 vs EAS (si les deux)
# =============================================================


# ======================== OPTIONS AVANCÉES ===================
SAVE_CHECKPOINTS = False    # Sauvegarder des checkpoints intermédiaires
CHECKPOINT_FREQ = 100000    # Fréquence de sauvegarde (timesteps)
VERBOSE = True              # Affichage détaillé pendant l'entraînement
# =============================================================


def validate_config():
    """Valide la configuration"""
    if not TRAIN_TD3 and not TRAIN_EAS_TD3:
        raise ValueError(
            "❌ Erreur de configuration!\n"
            "Au moins un algorithme doit être activé.\n"
            "Activez TRAIN_TD3 ou TRAIN_EAS_TD3 (ou les deux) dans config.py"
        )
    
    if MAX_TIMESTEPS < START_TIMESTEPS:
        raise ValueError(
            f"❌ MAX_TIMESTEPS ({MAX_TIMESTEPS}) doit être >= "
            f"START_TIMESTEPS ({START_TIMESTEPS})"
        )
    
    return True


def print_config():
    """Affiche la configuration actuelle"""
    print("\n" + "="*70)
    print("📋 CONFIGURATION")
    print("="*70)
    print(f"\n🎮 Environnement:")
    print(f"   {ENV_NAME}")
    print(f"\n🤖 Algorithmes à entraîner:")
    print(f"   TD3:     {'✓' if TRAIN_TD3 else '✗'}")
    print(f"   EAS-TD3: {'✓' if TRAIN_EAS_TD3 else '✗'}")
    print(f"\n⏱️  Entraînement:")
    print(f"   Max timesteps: {MAX_TIMESTEPS:,} par algorithme")
    print(f"   Eval frequency: {EVAL_FREQ:,}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Seed: {SEED}")
    
    if TRAIN_EAS_TD3:
        print(f"\n🧬 EAS (PSO):")
        print(f"   Population: {PSO_POP_SIZE}")
        print(f"   Iterations: {PSO_ITERATIONS}")
        print(f"   ω={PSO_OMEGA}, c1={PSO_C1}, c2={PSO_C2}")
    
    print(f"\n💾 Sauvegarde:")
    print(f"   Models: {SAVE_DIR}/")
    print(f"   Logs: {LOG_DIR}/")
    print(f"   Figures: {FIG_DIR}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    validate_config()
    print_config()
