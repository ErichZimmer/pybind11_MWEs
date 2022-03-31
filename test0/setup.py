import sys
from glob import glob

from pybind11 import get_cmake_dir
# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"


ext_modules = [
    Pybind11Extension("example_filters",
        sorted(glob("src/*.cpp")),
        #include_dirs=sorted(glob("src/*.h")),
        ),
]

setup(
    name="example_filters",
    version=__version__,
    author="Erich Zimmer",
    author_email="erich_zimmer@hotmail.com",
    url="",
    description="A test project using pybind11",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)