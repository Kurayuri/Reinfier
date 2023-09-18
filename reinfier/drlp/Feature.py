class Dynamic:
    def __init__(self, lower=None, upper=None, lower_closed = True, upper_closed = True, 
                 lower_rho=None, upper_rho=None, weight=None):
        self.lower = lower
        self.upper = upper
        self.lower_rho = lower_rho
        self.upper_rho = upper_rho
        self.weight = weight
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed

    def __repr__(self) -> str:
        return f'Dynamic({self.lower}, {self.upper})'
        return f'Dynamic({self.lower}, {self.upper}, {self.lower_closed}, {self.upper_closed}, {self.lower_rho}, {self.upper_rho}, {self.weight})'


class Static:
    def __init__(self, lower=None, upper=None,lower_closed = True, upper_closed = True,):
        self.lower = lower
        self.upper = upper
        self.lower_closed = lower_closed
        self.upper_closed = upper_closed

    def __repr__(self) -> str:
        return f'Static({self.lower}, {self.upper})'

        return f'Static({self.lower}, {self.upper}, {self.lower_closed}, {self.upper_closed})'
