from math import dist
from typing import Dict, List
from .classifier_interface import ClassifierInterface
from src.datasets.dataset_interface import DatasetInterface
from collections import Counter


class KnnClassifier(ClassifierInterface):
    def __init__(self) -> None:
        super().__init__()
        self.k = 5                                              #olha os 5 vizinhos mais proximos

    def train(self, train_dataset: DatasetInterface) -> None:
        # salvar as amostras do dataset
        self.dict_class = {}                                    #dicionario que relacionara cada classe com suas amostras

        for i in range (train_dataset.size()):                  #percorre todo o arquivo de treino
            amostra, classe = train_dataset.get(i)            #separa em amostra e classe o retorno do get (imagem vetorizada, classe)
            if classe not in self.dict_class.keys():            #se a classe ainda não for uma chave do dicionario,...
                self.dict_class[classe] = []                    #...cria a chave
            self.dict_class[classe].append(amostra)             #vai dando append das amostras

    def predict(self, test_dataset: DatasetInterface) -> List[str]:
        """ para cada amostra no dataset, buscar os k vizinhos mais proximos e 
        retornar a classe mais frequente entre eles """
        classes_preditas = []
        distancia1 = float("inf")
        distancia2 = float("inf")
        distancia3 = float("inf")
        distancia4 = float("inf")
        classe1 = 0
        classe2 = 0
        classe3 = 0
        classe4 = 0
        classe5 = 0

        for i in range (test_dataset.size()):
            vetor_teste, _ = test_dataset.get(i)
            for key in self.dict_class.keys():
                for vetor in self.dict_class[key]:
                    distancia = dist(vetor_teste, vetor)
                    if distancia < distancia1:
                        distancia1 = distancia
                        distancia2 = distancia1
                        distancia3 = distancia2
                        distancia4 = distancia3
                        classe1 = key
                        classe2 = classe1
                        classe3 = classe2
                        classe4 = classe3
                        classe5 = classe4
                    elif distancia < distancia2:
                        distancia2 = distancia
                        distancia3 = distancia2
                        distancia4 = distancia3
                        classe2 = key
                        classe3 = classe2
                        classe4 = classe3
                        classe5 = classe4
                    elif distancia < distancia3:
                        distancia3 = distancia
                        distancia4 = distancia3
                        classe3 = key
                        classe4 = classe3
                        classe5 = classe4
                    elif distancia < distancia4:
                        distancia4 = distancia
                        classe4 = key
                        classe5 = classe4
                cinco_classes = [classe1, classe2, classe3, classe4, classe5]
                contagem = Counter(cinco_classes)
                classe_mais_frequente = max(contagem, key=contagem.get)
                classes_preditas.append(classe_mais_frequente)
                distancia = 0
                distancia1 = 0
                distancia2 = 0
                distancia3 = 0
                distancia4 = 0
        return [classes_preditas]
