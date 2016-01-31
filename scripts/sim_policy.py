from rllab.sampler.utils import rollout
import argparse
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Speedup')
    args = parser.parse_args()
    data = joblib.load(args.file)
    policy = data['policy']
    mdp = data['mdp']
    mdp.start_viewer()
    path = rollout(mdp, policy, max_length=args.max_length, animated=True, speedup=args.speedup)
    mdp.stop_viewer()
    print 'Total reward: ', sum(path["rewards"])
