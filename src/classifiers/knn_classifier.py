from math import dist
from typing import Dict, List
from .classifier_interface import ClassifierInterface
from src.datasets.dataset_interface import DatasetInterface


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
        classificacao = []

        for idx_vetor in range (test_dataset.size()):
            for chave, lista_amostras in self.dict_class.items():
                for amostra in lista_amostras:
                    vetor_teste, _ = test_dataset.get(idx_vetor)
                    distancia = dist(vetor_teste, amostra)
                    classificacao.append((distancia, chave))

            # for key in range (self.dict_class.keys()):
            #     for amostra in range (self.dict_class[key])
            #     dist_aux = dist()


        return []
