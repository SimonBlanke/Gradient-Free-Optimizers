import os
import re

from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# Read the requirements
with open("requirements.txt", encoding="utf8") as f:
    requirements = f.readlines()


def find_version(*filepath):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name="gradient-free-optimizers",
    version=find_version("gradient_free_optimizers/__init__.py"),
    author="Simon Blanke",
    author_email="simon.blanke@yahoo.com",
    license="MIT",
    description=" ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["optimization"],
    url="https://github.com/SimonBlanke/Gradient-Free-Optimizers",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
    ],
    install_requires=requirements,
)
