from corr.annotated import load_annotated_index

__version__ = '0.0.1'
if __name__ == '__main__':
    df = load_annotated_index()
    print(df)
