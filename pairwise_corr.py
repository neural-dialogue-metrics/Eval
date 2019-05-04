from eval.correlate import load_dists_from_dir, DistGroup
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix')
    args = parser.parse_args()

    dists = load_dists_from_dir(args.prefix)
    group = DistGroup(dists, 'model_dataset')
    pw_corr = group['hred', 'ubuntu']
    print(pw_corr.df)
    print(pw_corr.corr)
