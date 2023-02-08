import json
from typing import Dict


def load_config(path: str) -> Dict:
    """ le o arquivo json e retorna como um dicionario """
    with open(path) as json_file:
        return json.load(json_file)

    # we're returning dummy values just to make the main program runnable
    # dummy_dict = {
    #     "train_dataset": {
    #         "type": "image",
    #         "path": "data/datasets/img_small/train.txt"
    #     },
    #     "test_dataset": {
    #         "type": "image",
    #         "path": "data/datasets/img_small/test.txt"
    #     },
    #     "classifier": {
    #         "type": "nc"
    #     }
    # }

    return dummy_dict
