import contextlib
import sys

class MultiOut(object):
    statt=0
    def __init__(self, *args):
        self.handles = args
    def write(self, s):
        for f in self.handles:
            f.write(s)

@contextlib.contextmanager
def output_wrapper(args):
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    sys.stdout = MultiOut(sys.stdout,args)
    sys.stderr = None
    yield
    sys.stdout = save_stdout
    sys.stderr = save_stderr