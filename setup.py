"""Setup script for blurgen package."""

from setuptools import setup, find_packages
from pathlib import Path

readme = Path(__file__).parent / "README.md"
long_description = readme.read_text() if readme.exists() else ""

setup(
    name="PC-TABD",
    version="0.1.0",
    author="Nikita Alutis",
    description="PC-TABD: A Physically-Grounded Dataset and Framework for Trajectory-Aware Motion Blur Synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/illaitar/PCTABD",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "scipy>=1.7.0",
        "pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    entry_points={
        "console_scripts": [
            "blurgen-csv=blurgen.cli.generate_csv:main",
            "blurgen-dataset=blurgen.cli.generate_dataset:main",
        ],
    },
)
