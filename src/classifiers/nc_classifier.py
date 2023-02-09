from math import sqrt, dist
from typing import Dict, List
from .classifier_interface import ClassifierInterface
from src.datasets.dataset_interface import DatasetInterface


class NearestCentroidClassifier(ClassifierInterface):
    def __init__(self) -> None:
        super().__init__()

    def train(self, train_dataset: DatasetInterface) -> None:
        self.dict_class = {}                                    #dicionario que relacionara cada classe com suas amostras

        for i in range (train_dataset.size()):                  #percorre todo o arquivo de treino
            amostra, classe = train_dataset.get(i)            #separa em amostra e classe o retorno do get (imagem vetorizada, classe)
            if classe not in self.dict_class.keys():            #se a classe ainda não for uma chave do dicionario,...
                self.dict_class[classe] = []                    #...cria a chave
            self.dict_class[classe].append(amostra)             #vai dando append das amostras 
        self.dict_centroides = {}

        for key, vetores in self.dict_class.items():
            centroide = [0] * len(vetores[0])
            for vetor in vetores:
                for coord in vetor:
                    centroide[coord] += vetor[coord]
            centroide = [coordinate / len(vetores) for coordinate in centroide]
            self.dict_centroides[key] = centroide




    def predict(self, test_dataset: DatasetInterface) -> List[str]:
        """ para cada amostra no dataset, buscar o centroide mais proximo e respectiva retornar a classe """
        classes_preditas = []

        for i in range (test_dataset.size()):
            vetor_teste, _ = test_dataset.get(i)                     #pego cada vetor de teste
            menor_distancia = float("inf")                           #comeco com uma distancia grande, a ideia eh substituir aqui a menor distancia
            for j in self.dict_centroides.keys():                    #percorro os centroides
                distancia=dist(vetor_teste, self.dict_centroides[j]) #calculo a distanca do vetor pro centroide
                if distancia < menor_distancia:                      #se a distancia for a menor que as analisadas anteriormente
                    classe = j                                       #salvo o indice
                    menor_distancia = distancia                      #salvo a menor distancia
            classes_preditas.append(classe)
        
        return classes_preditas                                      #retorno a lista de classes preditas

