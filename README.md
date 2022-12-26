# curvelops

Python wrapper for [CurveLab](http://www.curvelet.org)'s 2D and 3D curvelet transforms. It uses the [PyLops](https://pylops.readthedocs.io/) design framework to provide the forward and inverse curvelet transforms as matrix-free linear operations. If you are still confused, check out [some examples](https://github.com/PyLops/curvelops/tree/main/examples) below or the [PyLops website](https://pylops.readthedocs.io/)!

## Installation

Installing `curvelops` requires the following components:

- [FFTW](http://www.fftw.org/download.html) 2.1.5
- [CurveLab](http://curvelet.org/software.html) >= 2.0.2

Both of these packages _must be installed manually_. See more information below.
After these are installed, you may install `curvelops` with:

```bash
export FFTW=/path/to/fftw
export FDCT=/path/to/CurveLab
python3 -m pip install git+https://github.com/PyLops/curvelops@main
```

as long as you are using a `pip>=10.0`. To check, run `python3 -m pip --version`.

## Getting Started

For a 2D transform, you can get started with:

```python
import numpy as np
import curvelops as cl

x = np.random.normal(0., 1., (100, 50))
FDCT = cl.FDCT2D(dims=(100, 50))
c = FDCT * x.ravel()
xinv = FDCT.H * c
assert np.allclose(x, xinv.reshape(100, 50))
```

An excellent place to see how to use the library is the `examples/` folder. `Demo_Single_Curvelet` for example contains a `curvelops` version of the CurveLab Matlab demo.

![Demo](https://github.com/PyLops/curvelops/raw/main/docs/source/static/demo.png)
![Reconstruction](https://github.com/PyLops/curvelops/raw/main/docs/source/static/reconstruction.png)

## Tips and Tricks for Dependencies

### FFTW

For FFTW 2.1.5, you must compile with position-independent code support. Do that with

```bash
./configure --with-pic --prefix=/home/user/opt/fftw-2.1.5 --with-gcc=/usr/bin/gcc
```

The `--prefix` and `--with-gcc` are optional and determine where it will install FFTW and where to find the GCC compiler, respectively. We recommend using the same compile for FFTW, CurveLab and `curvelops`.

### CurveLab

In the file `makefile.opt` set `FFTW_DIR`, `CC` and `CXX` variables as required in the instructions. To keep things consistent, set `FFTW_DIR=/home/user/opt/fftw-2.1.5` (or whatever directory was used in the `--prefix` option). For the others, use the same compiler which was used to compile FFTW.

### curvelops

The `FFTW` variable is the same as `FFTW_DIR` as provided in the CurveLab installation. The `FDCT` variable points to the root of the CurveLab installation. It will be something like `/path/to/CurveLab-2.1.3` for the latest version.

## Useful links

* [Paul Goyes](https://github.com/PAULGOYES) has kindly contributed a rundown of how to install curvelops: [link to YouTube video (in Spanish)](https://www.youtube.com/watch?v=LAFkknyOpGY).

## Disclaimer

This package contains no CurveLab code apart from function calls. It is provided to simplify the use of CurveLab in a Python environment. Please ensure you own a CurveLab license as per required by the authors. See the [CurveLab website](http://curvelet.org/software.html) for more information. All CurveLab rights are reserved to Emmanuel Candes, Laurent Demanet, David Donoho and Lexing Ying.
