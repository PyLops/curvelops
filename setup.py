import os
import sys

from setuptools import find_packages, setup

if "clean" in sys.argv:
    from pathlib import Path

    # Delete any previously compiled files in pygeos
    p = Path("curvelops")
    for filename in p.glob("*.so"):
        print("removing '{}'".format(filename))
        filename.unlink()

from pybind11.setup_helpers import Pybind11Extension, build_ext

NAME = "curvelops"
AUTHOR = "Carlos Alberto da Costa Filho"
AUTHOR_EMAIL = "c.dacostaf@gmail.com"
URL = "https://github.com/PyLops/curvelops"
DESCRIPTION = "Python wrapper for CurveLab's 2D and 3D curvelet transforms"
LICENSE = "MIT"

with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

try:
    FFTW = os.environ["FFTW"]
except KeyError:
    print(
        """
    ==============================================================

    Please ensure the FFTW environment variable is set to the root
    of the FFTW 2.1.5 installation directory.

    ==============================================================
    """
    )
try:
    FDCT = os.environ["FDCT"]
except KeyError:
    print(
        """
    ==============================================================

    Please ensure the FDCT environment variable is set to the root
    of the CurveLab installation directory.

    ==============================================================
    """
    )


ext_modules = [
    Pybind11Extension(
        "fdct2d_wrapper",
        [os.path.join("cpp", "fdct2d_wrapper.cpp")],
        include_dirs=[
            os.path.join(FFTW, "fftw"),
            os.path.join(FDCT, "fdct_wrapping_cpp", "src"),
        ],
        libraries=["fftw"],
        library_dirs=[os.path.join(FFTW, "fftw", ".libs")],
        extra_objects=[
            os.path.join(FDCT, "fdct_wrapping_cpp", "src", "libfdct_wrapping.a")
        ],
        language="c++",
    ),
    Pybind11Extension(
        "fdct3d_wrapper",
        [os.path.join("cpp", "fdct3d_wrapper.cpp")],
        include_dirs=[
            os.path.join(FFTW, "fftw"),
            os.path.join(FDCT, "fdct3d", "src"),
        ],
        libraries=["fftw"],
        library_dirs=[os.path.join(FFTW, "fftw", ".libs")],
        extra_objects=[os.path.join(FDCT, "fdct3d", "src", "libfdct3d.a")],
        language="c++",
    ),
]

# Remove -stdlib=libc++ from MACOS flags if MACOS_GCC flag is equal to 1
# (This is required because pybind11 assumes OSX will use clang compiler but
# FFTW and FDCT may require switching to a gcc compiler in some OSX versions.
MACOS = sys.platform.startswith("darwin")
if MACOS and int(os.getenv("MACOS_GCC", 0)) == 1:
    for ext in ext_modules:
        new_flags = []
        for flag in ext.extra_compile_args:
            if flag != "-stdlib=libc++":
                new_flags.append(flag)
        ext.extra_compile_args = new_flags

        new_flags = []
        for flag in ext.extra_link_args:
            if flag != "-stdlib=libc++":
                new_flags.append(flag)
        ext.extra_link_args = new_flags

setup(
    name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    zip_safe=False,
    include_package_data=True,
    cmdclass={"build_ext": build_ext},
    ext_package="curvelops",
    ext_modules=ext_modules,
    packages=find_packages(exclude=["pytests"]),
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.9.1; python_version >= '3.9'",
        "pylops>=2.0",
        "matplotlib",
    ],
    setup_requires=[
        "pybind11>=2.6.0; python_version < '3.10'",
        "pybind11>=2.10.0; python_version >= '3.11'",
        "setuptools_scm",
    ],
    use_scm_version=dict(
        root=".", relative_to=__file__, write_to=f"{NAME}/_version.py"
    ),
    license=LICENSE,
    test_suite="pytests",
    tests_require=["pytest"],
    extras_require={"dev": ["pytest"]},
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="curvelet curvelab pylops",
)
