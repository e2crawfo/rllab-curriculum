require_relative '../utils'

# checked

params = {
  mdp: {
    _name: "mujoco_1_22.hopper_mdp",
    ctrl_cost_coeff: 1e-1,
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "parallel.linear_feature_baseline",
  },
  exp_name: "hopper",
  algo: {
    _name: "parallel.trpo",
    whole_paths: true,
    batch_size: 10000,
    max_path_length: 500,
    n_itr: 500,
    step_size: 0.1,
    plot: true,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: 1,
  plot: true,
}
command = to_command(params)
puts command
system(command)
