from pathlib import Path
import argparse
from corr.utils import UtterScoreDist
import logging
import pprint as pp

METRIC_NAME = 'serban_ppl'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Report system PPL from score files')
    parser.add_argument('-p', '--prefix', help='the location of the PPL score files')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    files = Path(args.prefix).glob(f'*{METRIC_NAME}.json')
    files = list(files)
    logging.info('Found PPL score files:\n{}'.format(pp.pformat(files)))

    for file in files:
        score_dist = UtterScoreDist.from_json_file(file)
        print(f"""
        Model: {score_dist.model}
        Dataset: {score_dist.dataset}
        PPL: {score_dist.system}
        """)
