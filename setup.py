"""
setup.py — ResLocoTransformer project setup.

Install in editable mode (recommended for development):
    pip install -e .

This makes the `models`, `envs`, `torchrl`, and `scripts` packages importable
from anywhere without needing to manipulate sys.path manually.

PyTorch is intentionally omitted from install_requires — install it first with
the CUDA variant matching your hardware (see requirements.txt for instructions).
"""

from setuptools import setup, find_packages

setup(
    name="resloco_transformer",
    version="0.1.0",
    description="Standalone LocoAttnResTransformer + MuJoCo RL training framework",
    python_requires=">=3.8",
    packages=find_packages(
        exclude=["*.pyc", "__pycache__", "*.egg-info"]
    ),
    install_requires=[
        "numpy>=1.21.0",
        "gym>=0.21.0",
        "mujoco>=3.0.0",
        "tensorboardX>=2.6.0",
        "tabulate>=0.9.0",
        "toolz>=0.12.0",
        "posix_ipc>=1.1.0",
    ],
    extras_require={
        "plot": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
