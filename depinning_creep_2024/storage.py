"""
h5py 'extensions'.
"""

from typing import TypeVar

import h5py


def create_extendible(file: h5py.File, key: str, dtype, ndim: int = 1, **kwargs) -> h5py.Dataset:
    """
    Create extendible dataset.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param dtype: Data-type to use.
    :param ndim: Number of dimensions.
    :param kwargs: An optional dictionary with attributes.
    """

    if key in file:
        return file[key]

    shape = tuple(0 for i in range(ndim))
    maxshape = tuple(None for i in range(ndim))
    dset = file.create_dataset(key, shape, maxshape=maxshape, dtype=dtype)

    for attr in kwargs:
        dset.attrs[attr] = kwargs[attr]

    return dset


def dset_extendible1d(
    file: h5py.File, key: str, dtype, value: TypeVar("T"), **kwargs
) -> h5py.Dataset:
    """
    Create extendible 1d dataset and store the first value.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param dtype: Data-type to use.
    :param value: Value to write at index 0.
    :param kwargs: An optional dictionary with attributes.
    """

    dset = file.create_dataset(key, (1,), maxshape=(None,), dtype=dtype)
    dset[0] = value

    for attr in kwargs:
        dset.attrs[attr] = kwargs[attr]

    return dset


def dset_extend1d(file: h5py.File, key: str, i: int, value: TypeVar("T")):
    """
    Dump and auto-extend a 1d extendible dataset.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param i: Index to which to write.
    :param value: Value to write at index ``i``.
    """

    dset = file[key]
    if dset.size <= i:
        dset.resize((i + 1,))
    dset[i] = value


def dump_with_atttrs(file: h5py.File, key: str, data: TypeVar("T"), **kwargs):
    """
    Write dataset and an optional number of attributes.
    The attributes are stored based on the name that is used for the option.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param data: Data to write.
    """

    file[key] = data
    for attr in kwargs:
        file[key].attrs[attr] = kwargs[attr]


def dump_overwrite(file: h5py.File, key: str, data: TypeVar("T"), **kwargs):
    """
    Dump or overwrite data.

    :param file: Opened HDF5 file.
    :param key: Path to the dataset.
    :param data: Data to write.
    :param kwargs: An optional dictionary with attributes.
    """

    if key in file:
        file[key][...] = data
    else:
        file[key] = data

    for attr in kwargs:
        file[key].attrs[attr] = kwargs[attr]
