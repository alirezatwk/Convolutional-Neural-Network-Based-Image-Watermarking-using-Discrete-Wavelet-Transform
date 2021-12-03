from abc import ABC, abstractmethod
from typing import Tuple


class BaseModel(ABC):

    def __init__(self, image_size: Tuple[int], watermark_size: Tuple[int]):
        self.image_size = image_size
        self.watermark_size = watermark_size

    @abstractmethod
    def get_model(self):
        pass
