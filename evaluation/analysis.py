import pandas as pd
import pathlib

from evaluation.udc_bundle import SUMMARY_FILE

SUMMARY_PATH: pathlib.Path = pathlib.Path(__file__).parent.parent / 'results' / SUMMARY_FILE
assert SUMMARY_PATH.is_file()

if __name__ == '__main__':
    df = pd.read_csv(SUMMARY_PATH)
    print(df)
