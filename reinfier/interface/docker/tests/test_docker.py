import time
import os

from .. import docker

CONTAINER_NAME = 'verisig'
CONTAINER_NAME = 'dnnv'


def test_exec():
    content = str(time.time())

    _, proc = docker.exec(CONTAINER_NAME, f"echo {content}")
    for chunk in proc:
        stdout, stderr = chunk
        print(stdout, stderr)

    assert stdout == [content]
    assert stderr == []


def test_copy_in():
    content = str(time.time())

    filename = f"{content}.txt"
    src_dirpath = "/tmp"
    src_path = os.path.join(src_dirpath, filename)
    with open(src_path, 'w') as f:
        f.write(content)

    try:
        dst_dirpath = "/tmp"
        dst_path = os.path.join(dst_dirpath, filename)
        docker.copy_in(CONTAINER_NAME, src_path, dst_dirpath)

        _, proc = docker.exec(CONTAINER_NAME, f"cat {dst_path}")
        for chunk in proc:
            stdout, stderr = chunk
            print(stdout, stderr)

        docker.exec(CONTAINER_NAME, f"rm {dst_path}")
        assert stdout == [content]
        assert stderr == []

    except Exception as e:
        os.remove(src_path)
        raise e


def test_write_in():
    content = str(time.time())

    filename = f"{content}.txt"

    dst_dirpath = "/tmp"
    dst_path = os.path.join(dst_dirpath, filename)

    docker.write_in(CONTAINER_NAME, content, dst_path)

    _, proc = docker.exec(CONTAINER_NAME, f"cat {dst_path}")
    for chunk in proc:
        stdout, stderr = chunk
        print(stdout, stderr)

    docker.exec(CONTAINER_NAME, f"rm {dst_path}")
    assert stdout == [content]
    assert stderr == []


def test_copy_out():
    content = str(time.time())

    filename = f"{content}.txt"
    src_dirpath = os.path.curdir
    src_path = os.path.join(src_dirpath, filename)
    with open(src_path, 'wb') as f:
        content = content.encode()
        f.write(content)

    try:
        dst_dirpath = "/tmp"
        dst_path = os.path.join(dst_dirpath, filename)
        docker.copy_in(CONTAINER_NAME, src_path, dst_dirpath)

        os.remove(src_path)

        docker.copy_out(CONTAINER_NAME, dst_path, src_dirpath)
        with open(src_path, "rb") as f:
            new_content = f.read()

        os.remove(src_path)
        docker.exec(CONTAINER_NAME, f"rm {dst_path}")

        assert new_content == content
    except Exception as e:
        os.remove(src_path)
        raise e
