from typing import NamedTuple, Dict, IO
from .Feature import Feature
from .DRLP import DRLP
from .aliases import *

# class BatchConfigs(NamedTuple):


class Variable:
    def __init__(self, lower, upper, lower_closed: bool = True, upper_closed=True, default=None) -> None:
        pass


class BatchConfig:
    BINARY = "binary"
    LINEAR = "linear"
    ITERATIVE = "iterative"

    def __init__(self,
                 lower: float,
                 upper: float,
                 default: float = 0,
                 precise: float = 1e-2,
                 step_method: str = LINEAR,
                 skip: bool = False):
        self.lower = lower
        self.upper = upper
        self.default = default
        self.precise = precise
        self.step_method = step_method
        self.skip = skip

    def __repr__(self) -> str:
        return f'[{self.lower}, {self.upper}] #{self.step_method}'

    __str__ = __repr__


class VerificationAnswer(NamedTuple):
    result: bool
    depth: int
    violation: Array

    # # def __repr__(self):
    # #     return f'CustomPerson(n'
    # def __repr__(self) -> str:
    #     return f'????'


class Breakpoint:
    TRUE_TO_FLASE = 1
    FALSE_TO_TRUE = 2
    TRUE_TO_TRUE = 3
    FALSE_TO_FALSE = 4

    def __init__(self, property: DRLP, ans: VerificationAnswer, btype: bool | VerificationAnswer = None) -> None:
        self.property = property
        self.ans = ans
        self.btype = btype
        if isinstance(btype, VerificationAnswer):
            self.btype = self.calc_btype(btype, ans)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<{self.property.variables}, {self.ans}, {Breakpoint.vc(self.btype)}>'

    __str__ = __repr__

    @classmethod
    def calc_btype(cls, prev: VerificationAnswer, curr: VerificationAnswer):
        if prev.result == True:
            return cls.TRUE_TO_TRUE if curr.result == True else cls.TRUE_TO_FLASE
        else:
            return cls.FALSE_TO_TRUE if curr.result == True else cls.FALSE_TO_FALSE

    @classmethod
    def vc(cls, btype):
        match btype:
            case cls.TRUE_TO_FLASE:
                return "TRUE -> FLASE"
            case cls.FALSE_TO_TRUE:
                return "FALSE -> TRUE"
            case cls.TRUE_TO_TRUE:
                return "TRUE == TRUE"
            case cls.FALSE_TO_FALSE:
                return "FALSE == FALSE"

class PropertyFeatures(NamedTuple):
    input: Dict[int, Feature]
    output: Dict[int, Feature]


class WhichFtr(NamedTuple):
    io: str
    index: int
