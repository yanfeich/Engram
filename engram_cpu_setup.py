# setup_simple.py
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

ONEDNN_INCLUDE_PATH = "/opt/intel/oneapi/dnnl/latest/include"
ONEDNN_LIBRARY_PATH = "/opt/intel/oneapi/dnnl/latest/lib"

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
        include_dirs=[ONEDNN_INCLUDE_PATH, current_dir],
        library_dirs=[ONEDNN_LIBRARY_PATH],
        libraries=['dnnl'],
        extra_compile_args={
            'cxx': ['-O3', '-march=native', '-fopenmp', '-std=c++17', '-mavx512f', '-mavx512cd', '-mavx512bw', 
                    '-mavx512dq', '-mavx512vl', '-mfma', '-mamx-tile', '-mamx-int8', '-mamx-bf16', '-DUSE_AMX=1',
                    '-DDNNL_FOUND=1']
        },
        extra_link_args=['-fopenmp', '-ldnnl'],
    )
]

setup(
    name='engram_cpu',
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    install_requires=['torch'],
)