from urllib.parse import urljoin
from urllib.request import urlretrieve
import tempfile
import os
import gzip
import struct
import numpy as np
import array

data_url = 'http://yann.lecun.com/exdb/mnist/'


def prepare_file(file_name):
    url = urljoin(data_url, file_name)
    local_dir = tempfile.gettempdir()
    local_file = os.path.join(local_dir, file_name)

    urlretrieve(url, local_file)

    with gzip.open(local_file,'rb') as f:
        types = {0x08: 'B', 0x09: 'b', 0x0b: 'h', 0x0c: 'i', 0x0d: 'f', 0x0e: 'd'}
        head = f.read(4)
        zeros, data_type, num_dimensions = struct.unpack('>HBB', head)
        data_type = types[data_type]
        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions, f.read(4*num_dimensions))
        data = array.array(data_type, f.read())
        data.byteswap()
        tmp = np.array(data).reshape(dimension_sizes)

    return tmp


def train_images():
    return prepare_file('train-images-idx3-ubyte.gz')


def train_labels():
    return prepare_file('train-labels-idx1-ubyte.gz')


def test_images():
    return prepare_file('t10k-images-idx3-ubyte.gz')


def test_labels():
    return prepare_file('t10k-labels-idx1-ubyte.gz')