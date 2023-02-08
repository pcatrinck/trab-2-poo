
from typing import List


def accuracy(true_classes: List[str], predicted_classes: List) -> float:
    q_vet = len(predicted_classes)
    acertos = 0
    for i in range (q_vet):
        if (true_classes[i] == predicted_classes[i]):
            acertos += 1
    return acertos/len(predicted_classes)