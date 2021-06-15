from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cu_gemm',
    ext_modules=[
        CUDAExtension('cu_gemm', [
            'gemm.cpp',
            'gemm_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
