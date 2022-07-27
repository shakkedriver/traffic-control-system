from CarFactory import CarFactory
import random

from NormalCar import NormalCar


class NormalCarFactory(CarFactory):
    def __init__(self, path, creation_frequency=0.1):
        super().__init__(path)
        self.creation_frequency = creation_frequency

    def create(self):
        generate_car = random.random() < self.creation_frequency
        if not generate_car:
            return None
        else:
            speed = random.randint(5, 15)  # todo make this general
            car = NormalCar(self.path,0,speed)
            return car

