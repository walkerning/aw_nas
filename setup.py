# -*- coding: utf-8 -*-
import os
from setuptools import setup, find_packages

HERE = os.path.dirname(os.path.abspath((__file__)))

# meta infos
NAME = "aw_nas"
DESCRIPTION = "The nas implementation of our group."

with open(os.path.join(os.path.dirname(__file__), "aw_nas", "VERSION")) as f:
    VERSION = f.read().strip()


AUTHOR = "foxfi"
EMAIL = "foxdoraame@gmail.com"

# package contents
MODULES = []
PACKAGES = find_packages(exclude=["tests.*", "tests"])

# dependencies
INSTALL_REQUIRES = [
    "torch>=1.0.0", # torch
    "torchvision",  # torch
    "numpy",        # math lib
    "scipy",        # math lib
    "six",          # 2-3 compatability
    "PyYaml",       # config file parsing
    "click",        # command line interface
    "graphviz"      # visualize architecture dag
]

TESTS_REQUIRE = []

def read_long_description(filename):
    path = os.path.join(HERE, filename)
    if os.path.exists(path):
        return open(path).read()
    return ""

setup(
    name=NAME,
    version=VERSION,
    license="MIT",
    url="https://github.com/walkerning/aw_nas",
    author=AUTHOR,
    author_email=EMAIL,

    description=DESCRIPTION,
    long_description=read_long_description("README.md"),

    py_modules=MODULES,
    packages=PACKAGES,

    entry_points={
        "console_scripts": [
            "awnas=aw_nas.main:main"
        ]
    },
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,

    zip_safe=True,
    package_data={
        "aw_nas": ["VERSION"]
    }
)
