class SearchConfig:
    BINARY = "binary"
    LINEAR = "linear"
    ITERATIVE = "iterative"
    def __init__(self, lower:float, upper:float, precise:float = 1e-2, method: str = "binary" ):
        self.lower = lower
        self.upper = upper
        self.precise = precise
        self.method = method