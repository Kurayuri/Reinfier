def log_call(*args):
    with open("log.txt", 'a+') as f:
        args = [str(arg) for arg in args]
        f.write(" ".join(args) + "\n")