import sys
from glob import glob
from os.path import dirname, join

import pybind11
import tempfile

import setuptools
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

try:
    from setuptools.errors import CompileError, LinkError
except ImportError:
    from distutils.errors import CompileError, LinkError
    
__version__ = "0.0.3"

project_name = "piv_filters"
 
base_path = dirname(__file__)

with open(join(base_path, "README.md"), encoding = "utf-8") as f:
    long_desciption = f.read()

ext_modules = [
    Extension(
        project_name + "_core",
        sorted(glob(join(base_path, project_name+"/core/src", "*.cpp"))),
        include_dirs=[pybind11.get_include(), join(base_path, project_name+"/core/include")],
        language = "c++"
        ),
]

def has_flag(compiler, flagname):
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except CompileError:
            return False
    return True

def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag"""
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')
        
class BuildExt(build_ext):
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        opts = self.c_opts.get(compiler_type, [])
        if compiler_type == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
                
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)
            
    
setup(
    name=project_name,
    version=__version__,
    author="Erich Zimmer",
    author_email="erich_zimmer@hotmail.com",
    description="A test project using pybind11",
    long_description=long_desciption,
    long_description_content_type="text/markdown",
    include_package_data=True,
    setup_requires=[
        "setuptools",
    ],
    install_requires=[
        "numpy",
        "pybind11"
    ],
    packages=find_packages(where="./piv_filters/filters"),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)