from typing import Tuple, Any, Dict
from .dataset_interface import DatasetInterface

import cv2


class ImageDataset(DatasetInterface):
    def __init__(self, path: str) -> None:
        super().__init__(path)
        # ler arquivo contendo os nomes das imagens e as classes e armazenar
        # em uma lista
        path = 'data/datasets/img_small/test.txt'
        self.path = path
        self.images = []
        #self.lista_sumario = []

        with open (self.path) as file:
            self.linhas_sumario = file.readlines()
        self.path, _ = self.path.rsplit("/", 1)
        
    def size(self) -> int:
        # retornar tamanho do dataset (numero de linhas do arquivo)
        return len(self.linhas_sumario)

    def get(self, idx: int) -> Tuple[Any, str]:
        # ler a i-esima imagem do disco usando a biblioteca cv2 e retornar
        # a imagem e a respectiva classe
        self.image_path, self.image_class = self.linhas_sumario[idx].split()
        self.image_path = self.path + '/' + self.image_path
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        
        image_vector = []
        for i in range(len(image)):
            for j in range(len(image[0])):
                image_vector.append(int(image[i][j]))
        #return image_vector, str(self.image_class[idx])
        #tava dando problema de index out of range com o idx no image_class
        
        return image_vector, str(self.image_class)
