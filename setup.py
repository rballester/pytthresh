#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="pytthresh",
    version="0.1.0",
    description="Python Implementation of the TTHRESH Grid Data Compressor",
    long_description="",
    url="https://github.com/rballester/pytthresh",
    author="Rafael Ballester-Ripoll",
    author_email="rafael.ballester@ie.edu",
    packages=[
        "pytthresh",
    ],
    include_package_data=True,
    install_requires=[
        "constriction",
        "numba",
        "numpy",
        "pandas",
        "quimb",
        "scipy",
        "typer",
    ],
    license="LGPL",
    zip_safe=False,
    keywords="pytthresh",
    classifiers=[
        "License :: OSI Approved :: ISC License (ISCL)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    test_suite="tests",
    tests_require="pytest",
)
