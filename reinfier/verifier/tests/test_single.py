from ... import res
from .. import single

def test_verify():
    network, property = res.get_example()
    result, depth, violation  = single.verify(network, property)
    assert result == False
    assert depth == 2
