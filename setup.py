# -*- coding: utf-8 -*-
import os
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

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
    "torch>=1.0.0",        # torch
    "torchvision>=0.4.0",  # torchvision, for torchvision.ops.nms
    "numpy",               # math lib
    "scipy",               # math lib
    "six",                 # 2-3 compatability
    "PyYaml",              # config file parsing
    "click",               # command line interface
    "graphviz",            # visualize architecture dag
    # other utils
    "imageio",
    "setproctitle"
]

EXTRAS_REQUIRE = {
    "vis": ["tensorboardX<=1.6"],
    "det": ["opencv-python", "pycocotools", "torchvision>=0.4.0"]
}

TESTS_REQUIRE = [
    "pytest",
    "pytest-cov",
]

def read_long_description(filename):
    path = os.path.join(HERE, filename)
    if os.path.exists(path):
        return open(path).read()
    return ""

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ["tests/", "-x", "--cov"]
        self.test_suite = True

    def run_tests(self):
        # import here, cause outside the eggs aren"t loaded
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

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
            "awnas=aw_nas.main:main",
            "awnas-hw=aw_nas.main_hardware:main"
        ]
    },

    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    tests_require=TESTS_REQUIRE,

    cmdclass={"test": PyTest},

    zip_safe=True,
    package_data={
        "aw_nas": ["VERSION"]
    }
)
