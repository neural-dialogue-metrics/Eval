from corr.utils import DataIndex

if __name__ == '__main__':
    data_index = DataIndex('./save')
    for triple in data_index.iter_triples():
        print(triple)
