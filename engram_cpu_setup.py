# setup_simple.py
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义扩展
ext_modules = [
    cpp_extension.CppExtension(
        'engram_cpu',
        [
            'engram_cpu.cpp',
            'engram_cpu_binding.cpp'
        ],
        include_dirs=[current_dir],
        extra_compile_args={
            'cxx': ['-O3', '-march=native', '-fopenmp', '-std=c++17']
        },
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='engram_cpu',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=['torch'],
)