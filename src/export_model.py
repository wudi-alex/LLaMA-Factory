import os

os.environ['TRANSFORMERS_CACHE'] = '/datasets/Large_Language_Models'
from llmtuner import export_model


def main():
    export_model()


if __name__ == "__main__":
    main()
