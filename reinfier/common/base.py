from abc import ABC, abstractmethod
import os
import enum
import typing

from typing import IO, Literal, Union

PathLike = IO | str | os.PathLike


def _load_bytes(f: PathLike) -> bytes:
    if hasattr(f, "read") and callable(typing.cast(IO[bytes], f).read):
        content = typing.cast(IO[bytes], f).read()
    else:
        f = typing.cast(Union[str, os.PathLike], f)
        with open(f, "rb") as readable:
            content = readable.read()
    return content


def _save_bytes(content: bytes, f: IO[bytes] | str | os.PathLike) -> None:
    if hasattr(f, "write") and callable(typing.cast(IO[bytes], f).write):
        typing.cast(IO[bytes], f).write(content)
    else:
        f = typing.cast(Union[str, os.PathLike], f)
        with open(f, "wb") as writable:
            writable.write(content)


def _load_texts(f: PathLike) -> bytes:
    if hasattr(f, "read") and callable(typing.cast(IO[bytes], f).read):
        content = typing.cast(IO[bytes], f).read()
    else:
        f = typing.cast(Union[str, os.PathLike], f)
        with open(f, "r") as readable:
            content = readable.read()
    return content


def _save_bytes(content: bytes, f: IO[bytes] | str | os.PathLike) -> None:
    if hasattr(f, "write") and callable(typing.cast(IO[bytes], f).write):
        typing.cast(IO[bytes], f).write(content)
    else:
        f = typing.cast(Union[str, os.PathLike], f)
        with open(f, "w") as writable:
            writable.write(content)


class BaseObject(ABC):
    def __init__(self, arg, filename):
        self.path = None
        self.obj = None

    def save(self, path: str | None = None):
        path = self.path if path is None else path
        self.save_obj(path)

    @abstractmethod
    def save_obj(self, path):
        pass

    def isValid(self):
        return not (self.path is None and self.obj is None)


class BaseEnum(enum.Enum):
    @property
    def raw_name(self):
        return self.name.lower()

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.name.lower() == value.lower():
                return member
        return super()._missing_(value)
