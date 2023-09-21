from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

extensions = [
    Extension(
        "translator_microphone",
        sources=["translator_microphone.pyx"],
        libraries=["python310"],  # 这里需要指定正确的Python库名称
        language='c',
        extra_compile_args=['-w']
    )
]

setup(
    ext_modules=cythonize(extensions),
)
