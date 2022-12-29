
from .import lib
from .DRLPTransformer import DRLPTransformer
from .DRLP import DRLP


def edit(drlp: str, code: str, to_overwrite: bool = False) -> str:
    if to_overwrite:
        return overwrite(drlp, code)
    else:
        return append(drlp, code)


def append(drlp: str, code: str) -> str:
    drlp_v, drlp_pq = lib.split_drlp_vpq(drlp)
    drlp_v += code
    return "\n".join((drlp_v, DRLPTransformer.PRECONDITION_DELIMITER, drlp_pq))


def overwrite(drlp: str, code: str) -> str:
    drlp_v, drlp_pq = lib.split_drlp_vpq(drlp)
    drlp_v = code
    return "\n".join((drlp_v, DRLPTransformer.PRECONDITION_DELIMITER, drlp_pq))
