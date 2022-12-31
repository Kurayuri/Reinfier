from .. import Setting


def is_branchable(verifier):
    return verifier in Setting.BranchableVerifiers
