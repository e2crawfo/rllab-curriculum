import os
import random

os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import sys
from multiprocessing import cpu_count
from rllab.misc.instrument import run_experiment_lite
from rllab.misc.instrument import VariantGenerator
from rllab import config

from curriculum.experiments.starts.maze.maze_ant.maze_ant_trpo_algo import run_task

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ec2', '-e', action='store_true', default=False, help="add flag to run in ec2")
    parser.add_argument('--clone', '-c', action='store_true', default=False,
                        help="add flag to copy file and checkout current")
    parser.add_argument('--local_docker', '-d', action='store_true', default=False,
                        help="add flag to run in local dock")
    parser.add_argument('--type', '-t', type=str, default='', help='set instance type')
    parser.add_argument('--price', '-p', type=str, default='', help='set betting price')
    parser.add_argument('--subnet', '-sn', type=str, default='', help='set subnet like us-west-1a')
    parser.add_argument('--name', '-n', type=str, default='', help='set exp prefix name and new file name')
    parser.add_argument('--debug', action='store_true', default=False, help="run code without multiprocessing")
    args = parser.parse_args()

    mode = 'local'
    n_parallel = cpu_count() if not args.debug else 1

    vg = VariantGenerator()
    vg.add('maze_id', [0])  # default is 0
    vg.add('start_size', [15])  # this is the ultimate start we care about: getting the pendulum upright
    vg.add('start_goal', [[0, 4, 0.55, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1,]])
    vg.add('start_range',
           lambda maze_id: [4] if maze_id == 0 else [7])  # this will be used also as bound of the state_space
    # vg.add('start_center', lambda maze_id: [(2, 2)] if maze_id == 0 else [(0, 0)])
    vg.add('start_center', lambda maze_id, start_size: [(2, 2)] if maze_id == 0 and start_size == 2
                                                else [(2, 2, 0, 0)] if maze_id == 0 and start_size == 4
                                                else [(0, 0)] if start_size == 2
                                                else [(0, 0, 0, 0)])
    vg.add('ultimate_goal', lambda maze_id: [(0, 4)] if maze_id == 0 else [(2, 4), (0, 0)] if maze_id == 12 else [(4, 4)])
    vg.add('goal_size', [2])  # this is the ultimate goal we care about: getting the pendulum upright
    vg.add('goal_range',
           lambda maze_id: [4] if maze_id == 0 else [7])
    vg.add('goal_center', lambda maze_id: [(2, 2)] if maze_id == 0 else [(0, 0)])
    vg.add('terminal_eps', [1.0])
    # brownian params
    vg.add('brownian_variance', [1])
    # vg.add('initial_brownian_horizon', [50])
    # vg.add('brownian_horizon', [300])
    vg.add('baseline', ["MLP"])
    # goal-algo params
    vg.add('min_reward', [0.1])
    vg.add('max_reward', [0.9])
    vg.add('distance_metric', ['L2'])
    vg.add('extend_dist_rew', [False])  # !!!!
    vg.add('inner_weight', [0]) #TODO: try different inner weights
    vg.add('goal_weight', lambda inner_weight: [1000] if inner_weight > 0 else [1])
    vg.add('regularize_starts', [0])

    # vg.add('persistence', [1])
    # vg.add('n_traj', [3])  # only for labeling and plotting (for now, later it will have to be equal to persistence!)
    # vg.add('filter_bad_starts', [True])
    # vg.add('sampling_res', [2])
    # vg.add('with_replacement', [True])
    # replay buffer
    vg.add('replay_buffer', [True])
    vg.add('coll_eps', [0.05]) # should try this
    vg.add('num_new_starts', [200])
    vg.add('num_old_starts', [100])
    vg.add('feasibility_path_length', [100])
    # sampling params
    vg.add('horizon', lambda maze_id: [2000] if maze_id == 0 else [500])
    vg.add('outer_iters', lambda maze_id: [2000] if maze_id == 0 else [1000])
    vg.add('inner_iters', [5])  # again we will have to divide/adjust the
    vg.add('pg_batch_size', [120000])
    # policy initialization
    vg.add('output_gain', [0.1])
    vg.add('policy_init_std', [1])
    vg.add('learn_std', [False]) #2
    vg.add('adaptive_std', [False])
    vg.add('discount', [0.995]) #1
    # vg.add('seed', [83, 93])
    vg.add('seed', [43, 13, 23, 33, 53, 63, 73])
    # vg.add('seed', range(100, 600, 100))
    # sweeping: horizon, seed, feasibility_path_length, pg_batch_size
    # possible important: learn_std

    exp_prefix = 'ant-startgen-trpo4'
    print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_prefix, vg.size))

    for vv in vg.variants():
        # run_task(vv)
        run_experiment_lite(
            # use_cloudpickle=False,
            stub_method_call=run_task,
            variant=vv,
            mode='local',
            n_parallel=2,
            # Only keep the snapshot parameters for the last iteration
            snapshot_mode="last",
            seed=vv['seed'],
            exp_prefix=exp_prefix,
            # exp_name=exp_name,
        )
        sys.exit()
