from ..drlp.DRLP import DRLP
from typing import Callable, Dict, List, Set, Union

def choose_curriculum():
    pass

class Reintrainer:
    '''
    Reintrainer: Property Training Framework for Reinforcement Learning
    '''

    def __init__(self, properties: List[DRLP],curriculum_chosen_func: Callable):
        self.properties = properties
        self.properties_apply = []
        self.curriculum_chosen_func = curriculum_chosen_func

    def train(self, round: int):
        for i in range(round):
            pass
