from corr.pairwise import load_filename_data

if __name__ == '__main__':
    df = load_filename_data('./save')
    print(df)
