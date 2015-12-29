require_relative './utils'

(1..10).each do |seed|
  (0.1..1.0).step(0.1).each do |quantile|
    quantile = quantile.round(2)
    params = {
      mdp: {
        _name: "cartpole_mdp",
      },
      normalize_mdp: nil,
      policy: {
        _name: "mean_std_nn_policy",
        hidden_sizes: [],
      },
      vf: {
        _name: "mujoco_value_function",
      },
      exp_name: "ppo_cartpole_quantile_#{quantile}_seed_#{seed}",
      algo: {
        _name: "ppo",
        binary_search_penalty: false,
        whole_paths: true,
        quantile: quantile,
        batch_size: (1000 / quantile).to_i,
        max_path_length: 100,
        n_itr: 50,
      },
      n_parallel: 1,
      snapshot_mode: "none",
      seed: seed,
    }
    command = to_command(params)
    puts command
    `#{command}`
  end
end
