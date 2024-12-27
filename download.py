import os
from urllib.request import urlretrieve
from urllib.parse import urljoin

def download_mnist(data_path: str):
    # Dataset download path
    mnist_url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'

    # Create a path for the data
    if os.path.isdir(data_path):
        pass
    else:
        os.mkdir(data_path)

    # Function to download data from official website and unzip
    # Checks if the data file already exists and downloads the data and unzips it only if it doesn't already exist
    def download_parse(fgz):
        if os.path.exists(os.path.join(data_path, fgz[:-3])):
            pass
        else:
            url = urljoin(mnist_url, fgz)
            filename = os.path.join(data_path, fgz)
            urlretrieve(url, filename)
            os.system('gunzip ' + filename)

    # Paths of train and test images and labels
    download_parse('train-images-idx3-ubyte.gz')
    download_parse('t10k-images-idx3-ubyte.gz')
    download_parse('train-labels-idx1-ubyte.gz')
    download_parse('t10k-labels-idx1-ubyte.gz')


# Get the current working directory
cwd = os.getcwd()

data_path = os.path.join(cwd, 'data')
download_mnist(data_path)
