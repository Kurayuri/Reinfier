from sympy import Interval, Complement, oo
from typing import *
SympyInterval = Interval


class Interval:
    def __init__(
        self,
        lower=None,
        upper=None,
        lower_closed=True,
        upper_closed=True,
    ):
        self.lower = lower
        self.upper = upper
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed

    def sympy(self):
        return SympyInterval(self.lower if self.lower is not None else -oo,
                             self.upper if self.upper is not None else oo)

    def __repr__(self) -> str:
        return f'Interval({self.lower}, {self.upper})'

    @classmethod
    def from_sympy(cls, sympyInterval: SympyInterval):
        return cls(sympyInterval.start if sympyInterval.start is not -oo else None,
                   sympyInterval.end if sympyInterval.end is not oo else None)


class Feature(Interval):

    def __repr__(self) -> str:
        return f'Feature({self.lower}, {self.upper})'


class Dynamic(Feature):
    def __init__(self, lower=None, upper=None, lower_closed=True, upper_closed=True,
                 lower_rho=None, upper_rho=None, weight=None):
        super().__init__(lower, upper, lower_closed, upper_closed)
        self.lower_rho = lower_rho
        self.upper_rho = upper_rho
        self.weight = weight

    def __repr__(self) -> str:
        return f'Dynamic({self.lower}, {self.upper})'
        return f'Dynamic({self.lower}, {self.upper}, {self.lower_closed}, {self.upper_closed}, {self.lower_rho}, {self.upper_rho}, {self.weight})'


class Static(Feature):
    def __init__(self, lower=None, upper=None, lower_closed=True, upper_closed=True):
        super().__init__(lower, upper, lower_closed, upper_closed)

    def __repr__(self) -> str:
        return f'Static({self.lower}, {self.upper})'

        return f'Static({self.lower}, {self.upper}, {self.lower_closed}, {self.upper_closed})'


def calc_complement(a: Interval):
    return calc_difference(Interval(None, None), a)


def calc_difference(a: Interval, b: Interval) -> List[Interval]:
    union = Complement(a.sympy(), b.sympy())
    return [
        Interval.from_sympy(sympyInterval)
        for sympyInterval in list(union.args)
    ]


def calc_complement_spaces(intervals: Iterable[Interval]) -> List[List[Interval]]:
    intervals = [interval for interval in intervals]
    dim = len(intervals)
    ans = []
    for i in range(0, dim):
        comp_intervals = calc_complement(intervals[i])
        if len(comp_intervals) == 0:
            return []

        for comp_interval in comp_intervals:
            comp = []
            for j in range(0, i):
                comp.append(intervals[j])
            comp.append(comp_interval)
            for j in range(i + 1, dim):
                comp.append(Interval(None, None))
            ans.append(comp)
    return ans
