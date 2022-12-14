from .DRLPTransformer import DRLPTransformer, DRLPParsingError
import re


def split_drlp_pq(drlp):
    try:
        drlp = re.split("%s[^\n]*\n" % (DRLPTransformer.expectation_delimiter), drlp)
        if len(drlp) != 2:
            raise Exception
    except Exception:
        raise DRLPParsingError("Invalid DRLP format")
    return drlp[0], drlp[1]


def split_drlp_vpq(drlp):
    try:
        drlp = re.split("%s[^\n]*\n" % (DRLPTransformer.precondition_delimiter), drlp)
        if len(drlp) != 2:
            drlp.insert(0, "")
    except Exception as e:
        raise DRLPParsingError("Invalid DRLP format")
    return drlp[0], drlp[1]
