from ... import test
from .. import solver
import pytest

def test_measure_sensitivity():
    network,__ = test.get_example()
    input = [1,2]
    index = 0
    lower = 1
    upper = 3
    precision = 1e-2
    (max_sum, max_val), (min_sum,min_val) = solver.measure_sensitivity(network=network,input=input,index = index,lower = lower, 
        upper = upper,precision = precision)
    
    assert (max_sum, max_val, min_sum, min_val) == pytest.approx((10.01 ,2.99, 0.0,1.0))
    
    input = [1,2]
    index = 0
    lower = 1
    upper = 1
    precision = 1e-2
    (max_sum, max_val), (min_sum,min_val) = solver.measure_sensitivity(network=network,input=input,index = index,lower = lower, 
        upper = upper,precision = precision)
    assert (max_sum, max_val, min_sum, min_val) == pytest.approx((7.0 ,1.0, 0.0, 1.0))
