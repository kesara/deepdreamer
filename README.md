# Deep Dreamer
Easy to configure Python program that make use of [Google's DeepDream](https://github.com/google/deepdream/)

## Requirements
* Python 2.7
* [NumPy](https://pypi.python.org/pypi/numpy)
* [SciPy](https://pypi.python.org/pypi/scipy/)
* [Pillow](https://pypi.python.org/pypi/Pillow/)
* [Caffe](http://caffe.berkeleyvision.org/)

## Installation
1. Install [NumPy](https://pypi.python.org/pypi/numpy), [SciPy](https://pypi.python.org/pypi/scipy/), [Pillow](https://pypi.python.org/pypi/Pillow/) and [Caffe](http://caffe.berkeleyvision.org/). *NumPy, SciPy & Pillow can be installed via PIP.*
2. Download **deploy.prototxt** from [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).
3. Add line `force_backward: true` to **deploy.prototxt** file.
4. Download **bvlc_googlenet.caffemodel** from [bvlc_googlenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet).
5. Make sure those two files are on root directory of DeepDreamer.

## Usage
`python deepdreamer.py image.jpg`

### Configuration options
```
usage: deepdreamer.py [-h] [--zoom {true,false}] [--scale SCALE]
                      [--dreams DREAMS] [--itern ITERN] [--octaves OCTAVES]
                      [--octave-scale OCTAVE_SCALE] [--layers LAYERS]
                      [--clip {true,false}] [--list-layers]
                      [image]

positional arguments:
  image

optional arguments:
  -h, --help            show this help message and exit
  --zoom {true,false}   zoom dreams (default: true)
  --scale SCALE         scale coefficient for zoom (default: 0.05)
  --dreams DREAMS       number of images (default: 100)
  --itern ITERN         dream iterations (default: 10)
  --octaves OCTAVES     dream octaves (default: 4)
  --octave-scale OCTAVE_SCALE
                        dream octave scale (default: 1.4)
  --layers LAYERS       dream layers (default: inception_4c/output)
  --clip {true,false}   clip dreams (default: true)
  --list-layers         list layers
```
