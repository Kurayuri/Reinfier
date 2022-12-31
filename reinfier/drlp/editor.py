
from .import lib
from .DRLPTransformer import DRLPTransformer
from .DRLP import DRLP
from typing import Union


def edit(property: str, code: str, to_overwrite: bool = False) -> str:
    if to_overwrite:
        return overwrite(property, code)
    else:
        return append(property, code)


def append(property: str, code: str) -> str:
    drlp_v, drlp_pq = lib.split_drlp_vpq(property)
    drlp_v += code
    return "\n".join((drlp_v, DRLPTransformer.PRECONDITION_DELIMITER, drlp_pq))


def overwrite(property: str, code: str) -> str:
    drlp_v, drlp_pq = lib.split_drlp_vpq(property)
    drlp_v = code
    return "\n".join((drlp_v, DRLPTransformer.PRECONDITION_DELIMITER, drlp_pq))
