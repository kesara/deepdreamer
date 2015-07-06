# Deep Dreamer
Python program that make use of [Google's DeepDream](https://github.com/google/deepdream/)

## Requirements
* Python 2.7
* [NumPy](https://pypi.python.org/pypi/numpy)
* [SciPy](https://pypi.python.org/pypi/scipy/)
* [Pillow](https://pypi.python.org/pypi/Pillow/0)
* [Caffe](http://caffe.berkeleyvision.org/)

## Installation
1. Install [NumPy](https://pypi.python.org/pypi/numpy), [SciPy](https://pypi.python.org/pypi/scipy/), [Pillow](https://pypi.python.org/pypi/Pillow/0) and [Caffe](http://caffe.berkeleyvision.org/). *NumPy, SciPy & Pillow can be installed via PIP.*
2. Download **deploy.prototxt** from [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).
3. Add line `force_backward: true` to **deploy.prototxt** file.
4. Download **bvlc_googlenet.caffemodel** from [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).
5. Make sure those two files are on root directory of DeepDreamer.

## Usage
