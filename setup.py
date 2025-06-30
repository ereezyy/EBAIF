#!/usr/bin/env python3
"""
Emergent Behavior AI Framework (EBAIF)
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ebaif",
    version="0.1.0",
    author="Eddy Woods",
    author_email="contact@ereezyy.dev",
    description="Revolutionary AI framework for emergent behavior in gaming and interactive environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ereezyy/EBAIF",
    project_urls={
        "Bug Tracker": "https://github.com/ereezyy/EBAIF/issues",
        "Documentation": "https://ebaif.readthedocs.io/",
        "Source Code": "https://github.com/ereezyy/EBAIF",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "edge": [
            "onnx>=1.12.0",
            "onnxruntime>=1.12.0",
        ],
        "monitoring": [
            "wandb>=0.13.0",
            "tensorboard>=2.10.0",
            "prometheus-client>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ebaif=ebaif.cli:main",
            "ebaif-train=ebaif.training:main",
            "ebaif-serve=ebaif.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ebaif": [
            "data/*.json",
            "configs/*.yaml",
            "models/*.pt",
        ],
    },
    keywords=[
        "artificial intelligence",
        "emergent behavior",
        "neural architecture search",
        "distributed systems",
        "game ai",
        "multi-agent systems",
        "edge computing",
        "machine learning",
    ],
    zip_safe=False,
)

