import astpretty
import astor

def pprint(ast_root):
    astpretty.pprint(ast_root, show_offsets=False)

def print_source(ast_root):
    print(astor.to_source(ast_root))