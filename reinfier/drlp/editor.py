
from .DRLPTransformer import DRLPTransformer
from .import lib


def edit(drlp_vpq: str, code: str, to_overwrite: bool = False) -> str:
    if to_overwrite:
        return overwrite(drlp_vpq, code)
    else:
        return append(drlp_vpq, code)


def append(drlp_vpq: str, code: str) -> str:
    drlp_v, drlp_pq = lib.split_drlp_vpq(drlp_vpq)
    drlp_v += code
    return "\n".join((drlp_v, DRLPTransformer.PRECONDITION_DELIMITER, drlp_pq))


def overwrite(drlp_vpq: str, code: str) -> str:
    drlp_v, drlp_pq = lib.split_drlp_vpq(drlp_vpq)
    drlp_v = code
    return "\n".join((drlp_v, DRLPTransformer.PRECONDITION_DELIMITER, drlp_pq))
