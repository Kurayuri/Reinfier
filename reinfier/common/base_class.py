from abc import ABC, abstractmethod


class BaseObject(ABC):
    def __init__(self, arg, filename):
        self.path = None
        self.obj = None

    def save(self, path: str | None = None):
        path = self.path if path is None else path
        self.save_obj(path)

    @abstractmethod
    def save_obj(self, path):
        pass

    def isValid(self):
        return not (self.path is None and self.obj is None)
