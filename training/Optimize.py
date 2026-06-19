import optuna
import yaml
import numpy as np
import os
import json
from datetime import datetime
from training.train import train


def objective(trial):
    """
    Optuna objective function.
    Each call = one trial = one training run with suggested hyperparameters.
    Returns best average reward — Optuna maximizes this.
    """

    # --- Suggest hyperparameters ---
    # Optuna samples these intelligently using Bayesian optimization
    # It learns from previous trials which regions are promising

    lr_actor = trial.suggest_float('lr_actor', 1e-5, 5e-4, log=True)
    lr_critic = trial.suggest_float('lr_critic', 1e-5, 5e-4, log=True)
    clip_epsilon = trial.suggest_float('clip_epsilon', 0.05, 0.3)
    entropy_coef = trial.suggest_float('entropy_coef', 0.01, 0.1)
    update_epochs = trial.suggest_int('update_epochs', 2, 8)
    update_freq = trial.suggest_categorical('update_freq', [64, 128, 256, 512])
    gat_heads = trial.suggest_categorical('gat_heads', [2, 4, 8])
    gat_hidden_dim = trial.suggest_categorical('gat_hidden_dim', [32, 64, 128])

    print(f"\n{'='*60}")
    print(f"Trial {trial.number} hyperparameters:")
    print(f"  lr_actor:      {lr_actor:.6f}")
    print(f"  lr_critic:     {lr_critic:.6f}")
    print(f"  clip_epsilon:  {clip_epsilon:.3f}")
    print(f"  entropy_coef:  {entropy_coef:.3f}")
    print(f"  update_epochs: {update_epochs}")
    print(f"  update_freq:   {update_freq}")
    print(f"  gat_heads:     {gat_heads}")
    print(f"  gat_hidden_dim:{gat_hidden_dim}")
    print(f"{'='*60}\n")

    # build override config
    # format: "section.param" matches how train.py applies overrides
    override_config = {
        'mappo.lr_actor': lr_actor,
        'mappo.lr_critic': lr_critic,
        'mappo.clip_epsilon': clip_epsilon,
        'mappo.entropy_coef': entropy_coef,
        'mappo.update_epochs': update_epochs,
        'training.update_freq': update_freq,
        'model.gat_heads': gat_heads,
        'model.gat_hidden_dim': gat_hidden_dim,
    }

    trial_name = f"optuna-trial-{trial.number}"

    try:
        # run training with these hyperparameters
        # short run — 30 episodes is enough to compare trials
        config_path = r'C:\Users\user\traffic_marl\configs\config_optuna.yaml'
        best_reward = train(
    config_path=config_path,
    override_config=override_config,
    trial_name=trial_name
)
        return best_reward

    except Exception as e:
        import traceback
        print(f"Trial {trial.number} failed:")
        traceback.print_exc()
        return -9999.0


def optimize(n_trials=20):
    """
    Run Optuna hyperparameter search.

    Args:
        n_trials: number of trials to run
                  20 trials × ~30 min each = ~10 hours
    """

    print(f"\nStarting Optuna hyperparameter search")
    print(f"Trials: {n_trials}")
    print(f"Each trial: 30 episodes (short run for speed)")
    print(f"Objective: maximize average reward\n")

    # create study — maximize reward (it's negative so we maximize toward 0)
    study = optuna.create_study(
        direction='maximize',
        study_name='traffic-marl-gnn-hpo',
        storage='sqlite:///checkpoints/optuna_study.db',  # saves to disk
        load_if_exists=True    # resume if interrupted
    )

    # run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True
    )

    # --- Results ---
    print(f"\n{'='*60}")
    print(f"Optuna search complete!")
    print(f"Number of trials: {len(study.trials)}")
    print(f"\nBest trial:")
    print(f"  Reward: {study.best_value:.2f}")
    print(f"  Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")

    # save best hyperparameters to yaml
    _save_best_config(study.best_params)

    # print top 5 trials
    print(f"\nTop 5 trials:")
    sorted_trials = sorted(
        study.trials,
        key=lambda t: t.value if t.value is not None else -9999,
        reverse=True
    )
    for i, t in enumerate(sorted_trials[:5]):
        print(f"  {i+1}. Trial {t.number}: reward={t.value:.2f}")
        for k, v in t.params.items():
            print(f"       {k}: {v}")

    return study


def _save_best_config(best_params):
    """
    Save the best hyperparameters found by Optuna
    into a ready-to-use config file for final training.
    """
    os.makedirs('checkpoints', exist_ok=True)

    # load base config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # apply best params
    config['mappo']['lr_actor'] = best_params['lr_actor']
    config['mappo']['lr_critic'] = best_params['lr_critic']
    config['mappo']['clip_epsilon'] = best_params['clip_epsilon']
    config['mappo']['entropy_coef'] = best_params['entropy_coef']
    config['mappo']['update_epochs'] = best_params['update_epochs']
    config['training']['update_freq'] = best_params['update_freq']
    config['model']['gat_heads'] = best_params['gat_heads']
    config['model']['gat_hidden_dim'] = best_params['gat_hidden_dim']

    # set full training episodes for final run
    config['training']['episodes'] = 500

    # save as best config
    with open('configs/config_best.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\nBest config saved to configs/config_best.yaml")
    print(f"Use it for final training with:")
    print(f"  python -m training.train --config configs/config_best.yaml")


if __name__ == '__main__':
    optimize(n_trials=20)