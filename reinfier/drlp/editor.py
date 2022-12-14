
from .import lib
from .DRLPTransformer import DRLPTransformer


def edit(drlp, code, to_overwrite=False):
    if to_overwrite:
        return overwrite(drlp, code)
    else:
        return append(drlp, code)


def append(drlp, code):
    drlp_v, drlp_pq = lib.split_drlp_vpq(drlp)
    drlp_v += code
    return "\n".join((drlp_v, DRLPTransformer. precondition_delimiter, drlp_pq))


def overwrite(drlp, code):
    drlp_v, drlp_pq = lib.split_drlp_vpq(drlp)
    drlp_v = code
    return "\n".join((drlp_v, DRLPTransformer. precondition_delimiter, drlp_pq))
