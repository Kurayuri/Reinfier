from ..drlp.DRLP import DRLP
from typing import Callable, Dict, List, Set, Union
from bayes_opt import BayesianOptimization
from ..import alg
from ..import drlp
from ..import util


def choose_curriculum():
    pass


class Reintrainer:
    '''
    Reintrainer: Property Training Framework for Reinforcement Learning
    '''

    def __init__(self, properties: List[DRLP], curriculum_chosen_func: Callable,
                 model_path, verifier: str,
                 train_api: Callable, test_api: Callable
                 ):
        self.properties = properties
        self.properties_apply = []
        self.curriculum_chosen_func = curriculum_chosen_func
        self.verification_results = []
        self.verifier = verifier
        self.model_path = model_path

    def train(self, round: int):
        for i in range(round):
            optimizer = BayesianOptimization()
            optimizer.maximize()
            network = self.model_path

            for property in self.properties:
                ans = alg.verify(network, property, verifiver=self.verifier, to_induct=True)
                self.verification_results.append(ans)

    def exec_constraint(self, code, x, y):
        exec(code)
        if isinstance(code, str):
            del code
        return locals()

    def reward(self, x, y, reward: float):
        for property in self.properties:
            x = [x]
            y = [y]
            constraint = drlp.parse_drlp_get_constraint(property)
            util.log("## Constraint:\n", constraint.obj)
            util.log(constraint.obj)
            code = drlp.parse_constaint_to_code(constraint)
            violated = self.exec_constraint(code, x, y)[drlp.VIOLATED_ID]
            if violated:
                reward = -4.5
        return reward
