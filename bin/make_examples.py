import argparse
from pathlib import Path
from typing import Union, List

from eval.consts import SEPARATOR, CONTEXTS, REFERENCES, RESPONSES
from eval.loader import ResourceLoader
from eval.utils import UnderTest
from eval.models import Model
from eval.repo import all_models, get_dataset
from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)


def load_data_index():
    from eval.repo import all_datasets, all_models
    from eval.config_parser import parse_models_and_datasets



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Make examples by randomly sampling')
    parser.add_argument('-p', '--prefix', help='output prefix of the example files')
    parser.add_argument('-n', '--n-examples', type=int, default=15, help='number of examples to draw')
