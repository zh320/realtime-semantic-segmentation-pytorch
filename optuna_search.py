import os
import json
import optuna
from optuna.trial import TrialState
from optuna.storages import RetryFailedTrialCallback
import torch.distributed as dist
from core import SegTrainer
from configs.optuna_config import OptunaConfig

import warnings
warnings.filterwarnings("ignore")


class OptunaTrainer(SegTrainer):
    def __init__(self, config, trial):
        super().__init__(config)
        self.trial = trial

    def validate(self, config, *args, **kwargs):
        val_score = super().validate(config)
        self.after_validate(val_score)
        return val_score

    def after_validate(self, val_score):
        self.trial.report(val_score, self.cur_epoch)

        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()


if __name__ == '__main__':
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
    DDP = LOCAL_RANK != -1
    MAIN_RANK = LOCAL_RANK in [-1, 0]

    if DDP:
        dist.init_process_group(backend=dist.Backend.NCCL, init_method='env://')

    config = OptunaConfig()
    STUDY_NAME = config.study_name
    STUDY_DIRECTION = config.study_direction
    NUM_TRIAL = config.num_trial
    SAVE_DIR = config.save_dir
    del config

    trial_scores = {}
    def objective(trial):
        trial = optuna.integration.TorchDistributedTrial(trial) if DDP else trial

        config = OptunaConfig()
        config.init_dependent_config()

        if MAIN_RANK:
            print(f"Running trial: {trial.number}...\n")
        if config.save_every_trial:
            config.save_dir = f'{SAVE_DIR}/trial_{trial.number}'

        config.get_trial_params(trial)
        trainer = OptunaTrainer(config, trial)
        best_score = trainer.run(config)

        trial_scores[trial.number] = best_score.item()
        with open(f'{SAVE_DIR}/trial_scores.json', 'w') as f:
            json.dump(trial_scores, f, indent=1)

        return best_score

    if MAIN_RANK:
        storage = optuna.storages.RDBStorage("sqlite:///optuna.db", heartbeat_interval=1, failed_trial_callback=RetryFailedTrialCallback(),)
        study = optuna.create_study(storage=storage, study_name=STUDY_NAME, direction=STUDY_DIRECTION, load_if_exists=True)

        print('Using Optuna to perform hyperparameter search.\n')
        study.optimize(objective, n_trials=NUM_TRIAL, gc_after_trial=True)

        best_trial = study.best_trial
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        optuna_results = {'params':best_trial.params, 
                         'value':best_trial.value,
                         'finished_trials': len(study.trials), 
                         'pruned_trials': len(pruned_trials), 
                         'completed_trials': len(complete_trials)}

        with open(f'{SAVE_DIR}/optuna_results.json', 'w') as f:
            json.dump(optuna_results, f, indent=1)

    else:
        for _ in range(NUM_TRIAL):
            try:
                objective(None)
            except optuna.TrialPruned:
                pass