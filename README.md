# TD3 et EAS-TD3 pour Walker2d-v4

Implémentation et comparaison des algorithmes **TD3** (Twin Delayed Deep Deterministic Policy Gradient) et **EAS-TD3** (Evolutionary Action Selection TD3) pour l'environnement de locomotion bipède Walker2d-v4.

## Auteurs

- **Achraf Aitelquadi**
- **Sofiane Habach**
- **Mounir Lamsayah**

**Institution**: École Nationale Supérieure d'Informatique et d'Analyse des Systèmes (ENSIAS), Rabat, Maroc

---

## Description du Projet

Ce projet implémente et compare deux algorithmes d'apprentissage par renforcement profond pour la tâche de locomotion bipède dans l'environnement MuJoCo Walker2d-v4:

### TD3 (Twin Delayed DDPG)
Algorithme actor-critic off-policy qui améliore DDPG avec:
- **Clipped Double Q-Learning**: Deux réseaux critiques pour réduire la surestimation des Q-valeurs
- **Delayed Policy Updates**: Mise à jour de l'acteur moins fréquente pour stabiliser l'apprentissage
- **Target Policy Smoothing**: Ajout de bruit aux actions cibles pour lisser les estimations

### EAS-TD3 (Evolutionary Action Selection)
Extension de TD3 qui utilise l'optimisation par essaim particulaire (PSO) pour affiner les actions:
- Génère une population d'actions candidates
- Optimise via PSO guidé par les Q-valeurs du critique
- Sélectionne l'action avec la meilleure Q-valeur estimée
- Maintient une archive d'actions de haute qualité

### Environnement: Walker2d-v4
- **Objectif**: Faire marcher un robot bipède 2D le plus rapidement possible
- **État**: 17 dimensions (positions, angles, vitesses)
- **Actions**: 6 dimensions (couples articulaires continus dans [-1, 1])
- **Récompense**: `r = 1.0 + vitesse_avant - 0.001 × coût_contrôle`

---

## Installation

### Option 1: Installation Locale (Conda)

#### 1. Créer l'environnement Conda

```bash
# Créer un environnement Python 3.11
conda create -n RL python=3.11 -y
conda activate RL
```

#### 2. Installer les dépendances

```bash
# Installer PyTorch avec support CUDA (GPU recommandé)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

#### 3. Vérifier l'installation

```bash
# Vérifier CUDA
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}')"

# Vérifier Gymnasium et MuJoCo
python -c "import gymnasium as gym; env = gym.make('Walker2d-v4'); print('Walker2d-v4 OK')"
```

---

### Option 2: Docker (Recommandé pour Reproductibilité)

#### 1. Prérequis
- Docker Desktop installé
- NVIDIA GPU avec drivers (pour entraînement GPU)
- NVIDIA Container Toolkit (Linux) ou WSL2 + CUDA (Windows)

**Installation Docker**:
- Windows/Mac: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Linux: [Docker Engine](https://docs.docker.com/engine/install/)

**Installation NVIDIA Container Toolkit (Linux)**:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Construire l'image Docker

```bash
# Depuis le répertoire du projet
docker build -t walker2d-td3 .
```

**Temps de build**: ~10-15 minutes (télécharge 2.4 GB pour PyTorch)

#### 3. Lancer un conteneur

**Mode interactif (bash)**:
```bash
docker run --gpus all -it walker2d-td3 /bin/bash
```

**Entraînement automatique TD3** (500k timesteps):
```bash
docker run --gpus all walker2d-td3
```

**Entraînement avec paramètres personnalisés**:
```bash
docker run --gpus all walker2d-td3 python -m src.run_experiment --algorithm eas-td3 --timesteps 100000
```

**Avec sauvegarde des résultats sur l'hôte**:
```bash
# Windows PowerShell
docker run --gpus all -v ${PWD}:/workspace walker2d-td3

# Linux/Mac
docker run --gpus all -v $(pwd):/workspace walker2d-td3
```

---

## Utilisation

### Configuration

Modifiez les paramètres dans `src/config.py`:

```python
# Algorithmes à entraîner
TRAIN_TD3 = True           # Entraîner TD3
TRAIN_EAS_TD3 = False      # Entraîner EAS-TD3

# Paramètres d'entraînement
MAX_TIMESTEPS = 500000     # Durée totale
START_TIMESTEPS = 25000    # Exploration aléatoire initiale
BATCH_SIZE = 100           # Taille du batch
SEED = 0                   # Seed pour reproductibilité

# Hyperparamètres TD3
DISCOUNT = 0.99            # Facteur de discount γ
TAU = 0.005                # Soft update τ
POLICY_FREQ = 2            # Fréquence mise à jour acteur
LEARNING_RATE = 3e-4       # Learning rate

# Hyperparamètres EAS (PSO)
PSO_POP_SIZE = 10          # Taille population
PSO_ITERATIONS = 10        # Itérations PSO
PSO_OMEGA = 1.2            # Inertie ω
PSO_C1 = 1.5               # Coefficient cognitif c₁
PSO_C2 = 1.5               # Coefficient social c₂

# Niveau de verbosité (0-3)
VERBOSE = 2                # 0=silent, 1=minimal, 2=normal, 3=détaillé
```

### Entraînement

#### Commande de base
```bash
python -m src.run_experiment
```
Utilise les paramètres définis dans `config.py`

#### Avec arguments en ligne de commande
```bash
# Entraîner TD3 uniquement (500k timesteps)
python -m src.run_experiment --algorithm td3 --timesteps 500000

# Entraîner EAS-TD3
python -m src.run_experiment --algorithm eas-td3 --timesteps 500000

# Entraîner les deux algorithmes
python -m src.run_experiment --algorithms td3 eas-td3 --timesteps 500000

# Test rapide (10k timesteps, ~2 minutes)
python -m src.run_experiment --algorithm td3 --timesteps 10000

# Mode verbose (plus de détails)
python -m src.run_experiment --algorithm td3 --timesteps 100000 --verbose

# Avec seed personnalisé
python -m src.run_experiment --algorithm td3 --timesteps 500000 --seed 42
```

### Simulation et Évaluation

Après l'entraînement, utilisez `simulate.py` pour tester les modèles:

```bash
# Lister les modèles disponibles
python simulate.py --list-models

# Simuler avec rendu (10 épisodes)
python simulate.py --model models/TD3_Walker2d.pt --episodes 10

# Évaluation complète (100 épisodes, sans rendu)
python simulate.py --model models/TD3_Walker2d.pt --evaluate

# Comparer politique déterministe vs stochastique
python simulate.py --model models/EAS_TD3_Walker2d.pt --compare

# Sauvegarder une vidéo
python simulate.py --model models/TD3_Walker2d.pt --episodes 5 --save-video

# Sans rendu (évaluation rapide)
python simulate.py --model models/TD3_Walker2d.pt --episodes 50 --no-render
```

**Arguments disponibles**:
- `--model PATH`: Chemin vers le modèle (.pt)
- `--env ENV`: Environnement Gymnasium (défaut: Walker2d-v4)
- `--episodes N`: Nombre d'épisodes (défaut: 10)
- `--max-steps N`: Steps max par épisode (défaut: 1000)
- `--seed N`: Seed aléatoire (défaut: 0)
- `--no-render`: Désactiver le rendu
- `--stochastic`: Politique stochastique (avec bruit)
- `--save-video`: Sauvegarder vidéo dans `videos/`
- `--evaluate`: Évaluation complète (100 épisodes)
- `--compare`: Comparer déterministe vs stochastique

---

## Structure du Projet

```
robotic3/
├── src/                          # Code source principal
│   ├── algorithms/
│   │   ├── td3.py               # Implémentation TD3
│   │   ├── eas.py               # PSO et archive pour EAS
│   │   ├── networks.py          # Réseaux Actor/Critic
│   │   └── replay_buffer.py     # Buffer d'expérience
│   ├── environments/
│   │   └── walker_env.py        # Wrapper Walker2d
│   ├── utils/
│   │   ├── logger.py            # Logging et métriques
│   │   └── plotter.py           # Génération graphiques
│   ├── config.py                # Configuration centralisée
│   └── run_experiment.py        # Script d'entraînement
│
├── models/                       # Modèles sauvegardés (.pt)
├── results/
│   ├── logs/                    # Fichiers CSV de métriques
│   └── figures/                 # Graphiques générés
├── videos/                       # Vidéos de simulation
├── tests/                        # Tests unitaires
│   └── test_walker_env.py
│
├── docs/                         # Documentation
│   ├── algorithms.md            # Détails TD3 et EAS-TD3
│   ├── environment.md           # Spécifications Walker2d-v4
│   └── docker-usage.md          # Guide Docker complet
│
├── simulate.py                   # Script de simulation
├── requirements.txt              # Dépendances Python
├── Dockerfile                    # Configuration Docker
└── README.md                     # Ce fichier
```

---

## Résultats Attendus

### Performance (500k timesteps sur Kaggle 2×T4 GPU)

| Métrique | TD3 | EAS-TD3 |
|----------|-----|---------|
| Récompense finale | 3.33 ± 0.00 | 3.33 ± 0.00 |
| Temps d'entraînement | 1h 21min | 3h 52min |
| Vitesse (steps/sec) | 103 | 36 |
| Surcharge EAS | - | 2.9× |
| Taux acceptation évoluée | - | 23% |

### Fichiers générés

**Modèles** (`models/`):
- `TD3_Walker2d.pt`: Modèle TD3 entraîné
- `EAS_TD3_Walker2d.pt`: Modèle EAS-TD3 entraîné

**Logs CSV** (`results/logs/`):
- `episode_rewards.csv`: Récompenses par épisode
- `evaluations.csv`: Évaluations périodiques
- `training_losses.csv`: Pertes acteur/critique
- `eas_metrics.csv`: Métriques spécifiques EAS
- `action_magnitudes.csv`: Magnitudes des actions

**Figures** (`results/figures/`):
- Courbes d'apprentissage
- Analyse des pertes
- Évolution de l'exploration
- Métriques EAS (archive, taux acceptation)
- Comparaison TD3 vs EAS-TD3

---

## Paramètres Détaillés

### Hyperparamètres TD3

| Paramètre | Symbole | Valeur | Description |
|-----------|---------|--------|-------------|
| Max Timesteps | - | 500,000 | Durée totale entraînement |
| Start Timesteps | - | 25,000 | Exploration aléatoire initiale |
| Batch Size | \|B\| | 100 | Taille mini-batch |
| Discount | γ | 0.99 | Facteur discount futur |
| Tau | τ | 0.005 | Coefficient soft update |
| Policy Frequency | d | 2 | Fréquence mise à jour acteur |
| Learning Rate | α | 3×10⁻⁴ | Taux apprentissage Adam |
| Exploration Noise | σ | 0.1 | Écart-type bruit exploration |
| Target Noise | σ̃ | 0.2 | Écart-type target smoothing |
| Noise Clip | c | 0.5 | Clipping bruit cible |
| Replay Buffer | - | 1M | Capacité buffer |

### Hyperparamètres EAS (supplémentaires)

| Paramètre | Symbole | Valeur | Description |
|-----------|---------|--------|-------------|
| Swarm Size | N | 10 | Taille population PSO |
| PSO Iterations | T | 10 | Itérations évolution |
| Inertia Weight | ω | 1.2 | Poids inertie |
| Cognitive Weight | c₁ | 1.5 | Attraction personnel best |
| Social Weight | c₂ | 1.5 | Attraction global best |
| Max Velocity | v_max | 0.1 | Vitesse maximale particules |
| Archive Size | K | 2000 | Taille archive actions |

### Architecture Réseaux

**Acteur**: 17 → 400 → 300 → 6
- Entrée: 17 (état Walker2d)
- Couches cachées: [400, 300] (ReLU)
- Sortie: 6 (actions), Tanh

**Critique**: (17+6) → 400 → 300 → 1
- Entrée: 23 (état + action)
- Couches cachées: [400, 300] (ReLU)
- Sortie: 1 (Q-valeur)
- **Note**: Deux critiques identiques (double Q-learning)

---