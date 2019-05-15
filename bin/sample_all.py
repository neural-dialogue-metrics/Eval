from eval.repo import find_serban_models

if __name__ == '__main__':
    for model in find_serban_models():
        model.sample()
