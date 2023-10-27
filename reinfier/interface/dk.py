import tarfile
import docker
import io
import os

client = docker.from_env()
containers=client.containers

def exec(containor_name,cmd, to_split:bool=True):
    container = containers.get(containor_name)
    if not isinstance(cmd,str):
        cmd = " ".join(cmd)
    cmd = f'bash -c "{cmd}"'
    exit_code, proc = container.exec_run(cmd, user='root',stream=True,demux=True)

    def post_proc(proc):
        for chunk in proc:
            if to_split:
                stdout = [s for s in chunk[0].decode("utf-8").split("\n") if s] if chunk[0] else []
                stderr = [s for s in chunk[1].decode("utf-8").split("\n") if s] if chunk[1] else []
            else:
                stdout = chunk[0].decode("utf-8") if chunk[0] else chunk[0]
                stderr = chunk[1].decode("utf-8") if chunk[1] else chunk[1]
            yield stdout, stderr

    return exit_code, post_proc(proc)


def copy_in(containor_name, src_path, dst_dirpath):
    container = containers.get(containor_name)
    fileobj = io.BytesIO()
    with tarfile.open(fileobj=fileobj, mode='w') as tar:
        tar.add(src_path)
    container.put_archive(dst_dirpath, fileobj.getvalue())

def write_in(containor_name,content, dst_path):
    container = containers.get(containor_name)
    fileobj = io.BytesIO()
    with tarfile.open(fileobj=fileobj, mode='w') as tar:
        tar_info = tarfile.TarInfo(name=os.path.basename(dst_path))
        tar_info.size = len(content)
        tar.addfile(tar_info,io.BytesIO(content.encode('utf-8')))

    container.put_archive(os.path.dirname(dst_path), fileobj.getvalue())

def copy_out(containor_name,src,dst):
    pass

