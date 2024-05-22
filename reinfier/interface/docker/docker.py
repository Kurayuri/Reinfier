import tarfile
import docker
import io
import os

docker_containers = None


def init():
    if docker_containers is None:
        client = docker.from_env()
        docker_containers = client.containers
    return docker_containers


def exec(containor_name, cmd, to_decode: bool = True, to_split: bool = True):
    init()
    container = docker_containers.get(containor_name)
    if not isinstance(cmd, str):
        cmd = " ".join(cmd)
    cmd = f'bash -c "{cmd}"'
    exit_code, proc = container.exec_run(cmd, user='root', stream=True, demux=True)

    def post_proc(proc):
        for chunk in proc:
            if to_decode:
                if to_split:
                    stdout = [s for s in chunk[0].decode("utf-8").split("\n") if s] if chunk[0] else []
                    stderr = [s for s in chunk[1].decode("utf-8").split("\n") if s] if chunk[1] else []
                else:
                    stdout = chunk[0].decode("utf-8") if chunk[0] else chunk[0]
                    stderr = chunk[1].decode("utf-8") if chunk[1] else chunk[1]
            else:
                stdout, stderr = chunk
            yield stdout, stderr

    return exit_code, post_proc(proc)


def copy_in(containor_name, src_paths, dst_dirpath):
    init()

    if isinstance(src_paths, str):
        src_paths = [src_paths]
    container = docker_containers.get(containor_name)
    fileobj = io.BytesIO()
    with tarfile.open(fileobj=fileobj, mode='w') as tar:
        for src_path in src_paths:
            tar.add(src_path, arcname=os.path.basename(src_path))
    container.put_archive(dst_dirpath, fileobj.getvalue())


def write_in(containor_name, content, dst_path):
    init()

    container = docker_containers.get(containor_name)
    fileobj = io.BytesIO()
    with tarfile.open(fileobj=fileobj, mode='w') as tar:
        tar_info = tarfile.TarInfo(name=os.path.basename(dst_path))
        tar_info.size = len(content)
        tar.addfile(tar_info, io.BytesIO(content.encode('utf-8')))

    container.put_archive(os.path.dirname(dst_path), fileobj.getvalue())


def copy_out(containor_name, src_paths, dst_dirpath):
    init()

    if isinstance(src_paths, str):
        src_paths = [src_paths]

    container = docker_containers.get(containor_name)

    for src_path in src_paths:
        with open(os.path.join(dst_dirpath, os.path.basename(src_path)), "wb") as dst_file:
            cmd = f"cat {src_path}"
            exit_code, proc = container.exec_run(cmd, user='root', stream=True, demux=True)
            for chunk in proc:
                dst_file.write(chunk[0])
