.. _installation:

Installation
============

.. _requirements:

Requirements
------------

Installing Curvelops requires the following external components:

* `FFTW <https://www.fftw.org/download.html>`_  2.1.5
* `CurveLab <http://www.curvelet.org>`_ >= 2.0.2

Both of these packages must be installed manually.

Installing FFTW
~~~~~~~~~~~~~~~
Download and install with:


..  code-block:: console

    $ wget https://www.fftw.org/fftw-2.1.5.tar.gz
    $ tar xvzf fftw-2.1.5.tar.gz
    $ mkdir -p /home/$USER/opt/
    $ mv fftw-2.1.5/ /home/$USER/opt/
    $ cd /home/$USER/opt/fftw-2.1.5/
    $ ./configure --with-pic --prefix=/home/$USER/opt/fftw-2.1.5 --with-gcc=$(which gcc)
    $ make
    $ make install

The ``--prefix`` and ``--with-gcc`` are optional and determine where it will
install FFTW and where to find the GCC compiler, respectively. We recommend
using the same compiler for FFTW and CurveLab. To ensure that FFTW has been
installed correctly, run

..  code-block:: console

    $ make check


Installing CurveLab
~~~~~~~~~~~~~~~~~~~
After downloading the latest version of CurveLab, run

..  code-block:: console

    $ tar xvzf CurveLab-2.1.3.tar.gz
    $ mkdir -p /home/$USER/opt/
    $ mv CurveLab-2.1.3/ /home/$USER/opt/
    $ cd /home/$USER/opt/CurveLab-2.1.3/
    $ cp makefile.opt makefile.opt.bak

In the file ``makefile.opt`` set ``FFTW_DIR``, ``CC`` and ``CXX`` variables.
We recommend setting ``FFTW_DIR=/home/$USER/opt/fftw-2.1.5``
(or whatever directory was used in the ``--prefix`` option above), the output
of ``which gcc`` in CC (or whatever compiler was used in ``--with-gcc``), and
the ouput of ``which g++`` (or whatever C++ compiler is the equivalent of
the selected ``CC`` compiler). Once the variables are set in `makefile.opt`,
compile the library with

..  code-block:: console

    $ cd /home/$USER/opt/CurveLab-2.1.3/
    $ make clean
    $ make lib

To ensure that CurveLab is installed correctly, run

..  code-block:: console

    $ make test

Installing Curvelops
--------------------

Once FFTW and CurveLab are installed, install Curvelops with:

..  code-block:: console

    $ export FFTW=/path/to/fftw-2.1.5
    $ export FDCT=/path/to/CurveLab-2.1.3
    $ python3 -m pip install git+https://github.com/PyLops/curvelops@0.23

The ``FFTW`` variable is the same as ``FFTW_DIR`` as provided in the CurveLab
installation. The ``FDCT`` variable points to the root of the CurveLab
installation.
