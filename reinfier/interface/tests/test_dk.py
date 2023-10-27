import pytest
import time
import os

from ..import dk

CONTAINER_NAME = 'verisig'

def test_exec():
    content = str(time.time())

    _ , proc = dk.exec(CONTAINER_NAME,f"echo {content}")
    for chunk in proc:
        stdout, stderr = chunk
        print(stdout,stderr)


    assert stdout == [content]
    assert stderr == []

def test_copy_in():
    content = str(time.time())

    filename = f"{content}.txt"
    src_dirpath = os.path.curdir
    src_path = os.path.join(src_dirpath,filename)
    with open(src_path, 'w') as f:
        f.write(content)
    
    dst_dirpath = "/tmp"
    dst_path = os.path.join(dst_dirpath,filename)
    dk.copy_in(CONTAINER_NAME,src_path,dst_dirpath)

    _ , proc = dk.exec(CONTAINER_NAME, f"cat {dst_path}")
    for chunk in proc:
        stdout, stderr = chunk
        print(stdout,stderr)

    os.remove(src_path)
    assert stdout == [content]
    assert stderr == []
    dk.exec(CONTAINER_NAME,f"rm {dst_path}")

def test_write_in():
    content = str(time.time())

    filename = f"{content}.txt"
    
    dst_dirpath = "/tmp"
    dst_path = os.path.join(dst_dirpath,filename)

    dk.write_in(CONTAINER_NAME,content,dst_path)

    _ , proc = dk.exec(CONTAINER_NAME, f"cat {dst_path}")
    for chunk in proc:
        stdout, stderr = chunk
        print(stdout,stderr)
        
    assert stdout == [content]
    assert stderr == []
    dk.exec(CONTAINER_NAME,f"rm {dst_path}")