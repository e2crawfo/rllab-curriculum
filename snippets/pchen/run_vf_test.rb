require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    _name: "box2d.double_pendulum_mdp",
    # trig_angle: false,
    # frame_skip: 2,
  },
  # normalize_mdp: nil,
  policy: {
    _name: "mean_std_nn_policy",
    # hidden_layers: [],
  },
  vf: {
    # _name: "mujoco_value_function",
    _name: "nn_value_function",
    use_tr: true,
  },
  exp_name: "ppo_mc_seed_#{seed}",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 10,
    plot: true,
    step_size: 0.1,

  },
  n_parallel: 3,
  # snapshot_mode: "none",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)

