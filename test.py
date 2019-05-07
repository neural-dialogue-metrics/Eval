from corr.data import DataIndex

data_index = DataIndex('./save')
for triple in data_index.iter_triples:
    print(triple)
