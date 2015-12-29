require_relative './utils'

seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: nil,
  qf: {
    _name: "continuous_nn_q_function",
  },
  policy: {
    _name: "mean_nn_policy",
    hidden_layers: [],
    output_nl: 'lasagne.nonlinearities.tanh',
  },
  exp_name: "dpg_box2d_cartpole",
  algo: {
    _name: "dpg",
    batch_size: 32,
    n_epochs: 100,
    epoch_length: 1000,
    min_pool_size: 10000,
    replay_pool_size: 100000,
    discount: 0.99,
    qf_weight_decay: 0,
    max_path_length: 100,
    eval_samples: 10000,
    eval_whole_paths: true,
    renormalize_interval: 1000,
    # normalize_qval: false,
    policy_learning_rate: 1e-5,
  },
  es: {
    _name: "gaussian_strategy",
    max_sigma: 1,
    min_sigma: 0.01,
    sigma_decay_range: 200000,
  },
  #n_parallel: 1,
  #snapshot_mode: "none",
  seed: seed,
}
command = to_command(params)
puts command
system(command)

