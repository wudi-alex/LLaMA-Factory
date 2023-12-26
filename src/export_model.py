import os

os.environ['HF_HOME'] = '/datasets/Large_Language_Models'
from llmtuner import export_model


def main():
    export_model()


if __name__ == "__main__":
    main()
