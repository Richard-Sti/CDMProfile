from setuptools import setup, find_packages

setup(
    name="cdmprof",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
        "h5py",
        "sympy",
        "cffi",
        "tqdm",
    ],
    extras_require={
        "mpi": ["mpi4py"],
    },
)
