from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_data_loader(self):
        pass
