"""Default PPO settings for CPU-only training."""

config_ppo = {
    "max_iters": 400,
    # Core learning
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda_": 0.95,
    "train_batch_size_per_learner": 2048,
    "minibatch_size": 256,
    "num_epochs": 20,
    "entropy_coeff": 0.005,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.2,
    # Resources
    "num_learners": 1,
    "num_gpus_per_learner": 0,
    "num_cpus_per_learner": 1,
    "num_env_runners": 4,
    "num_envs_per_env_runner": 2,
    "num_cpus_per_env_runner": 1,
    "num_cpus_for_main_process": 1,
    "sample_timeout_s": 600,
    "rollout_fragment_length": "auto",
    # KL
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    # PBT defaults (used later by the PBT script)
    "pbt_lr_choices": [1e-4, 3e-4, 5e-4],
    "pbt_num_epochs_range": (10, 40),
    "pbt_minibatch_choices": [128, 256, 512],
    "pbt_train_batch_size_choices": [1024, 2048, 4096],
    "perturbation_interval": 5,
    "resample_probability": 0.25,
    "quantile_fraction": 0.25,
    "pbt_num_samples": 6,
}
