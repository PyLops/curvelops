[![Documentation](https://github.com/PyLops/curvelops/actions/workflows/pages/pages-build-deployment/badge.svg?branch=gh-pages)](https://pylops.github.io/curvelops/)
[![Slack Status](https://img.shields.io/badge/chat-slack-green.svg)](https://pylops.slack.com)

# curvelops

Python wrapper for [CurveLab](http://www.curvelet.org)'s 2D and 3D curvelet
transforms. It uses the [PyLops](https://pylops.readthedocs.io/) design
framework to provide the forward and inverse curvelet transforms as matrix-free
linear operations. If you are still confused, check out
[some examples](https://github.com/PyLops/curvelops/tree/main/examples) below
or the [PyLops website](https://pylops.readthedocs.io/)!

## Installation

Installing `curvelops` requires the following external components:

- [FFTW](http://www.fftw.org/download.html) 2.1.5
- [CurveLab](http://curvelet.org/software.html) >= 2.0.2

Both of these packages _must be installed manually_. See more information in
the [Documentation](https://pylops.github.io/curvelops/installation.html#requirements).
After these are installed, you may install `curvelops` with:

```bash
export FFTW=/path/to/fftw-2.1.5
export FDCT=/path/to/CurveLab-2.1.3
python3 -m pip install git+https://github.com/PyLops/curvelops@0.23.3
```

as long as you are using a `pip>=10.0`. To check, run `python3 -m pip --version`.

## Getting Started

For a 2D transform, you can get started with:

```python
import numpy as np
import curvelops as cl

x = np.random.randn(100, 50)
FDCT = cl.FDCT2D(dims=x.shape)
c = FDCT @ x
xinv = FDCT.H @ c
np.testing.assert_allclose(x, xinv)
```

An excellent place to see how to use the library is the
[Gallery](https://pylops.github.io/curvelops/gallery/index.html). You can also
find more examples in the
[`notebooks/`](https://github.com/PyLops/curvelops/tree/main/notebooks) folder.

![Demo](https://github.com/PyLops/curvelops/raw/main/docssrc/source/static/demo.png)
![Reconstruction](https://github.com/PyLops/curvelops/raw/main/docssrc/source/static/reconstruction.png)

## Useful links

* [Paul Goyes](https://github.com/PAULGOYES) has kindly contributed a rundown of how to install curvelops: [link to YouTube video (in Spanish)](https://www.youtube.com/watch?v=LAFkknyOpGY).

## Note

This package contains no CurveLab code apart from function calls. It is
provided to simplify the use of CurveLab in a Python environment. Please ensure
you own a CurveLab license as per required by the authors. See the
[CurveLab website](http://curvelet.org/software.html) for more information. All
CurveLab rights are reserved to Emmanuel Candes, Laurent Demanet, David Donoho
and Lexing Ying.
