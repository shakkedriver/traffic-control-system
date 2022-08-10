from abc import ABC, abstractmethod

class CarFactory(ABC):
    """
    this class is used to create cars in a certain path. each time we will call the create_car abstract method it will
    return a new car or None
    """
    def __init__(self,path):
        self.path = path

    @abstractmethod
    def create_car(self, env):
        """
        :return: a new car object or None if no car should be created
        """
        pass
