from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension("veccs.maxmin_cpp", ["src/_src/maxmin.cpp"]),
    Pybind11Extension("veccs.maxmin_var_cpp", ["src/_src/maxmin_var.cpp"]),
]

setup(cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
