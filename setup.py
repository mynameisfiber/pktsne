#!/usr/bin/env python

from setuptools import setup, find_packages
from pktsne import __version__

requirements = [line.strip() for line in open("requirements.txt")]

setup(
    name='PKTSNE',
    version=__version__,
    description=('Parametric T-SNE in Keras with support for data '
                 'supplied as a generator'),
    author='Micha Gorelick',
    author_email='mynameisfiber@gmail.com',
    url='http://github.com/mynameisfiber/pktsne/',
    download_url='https://github.com/mynameisfiber/pktsne/tarball/master',
    license="GNU Lesser General Public License v3 or later (LGPLv3+)",

    packages=find_packages(),
    install_requires=requirements,
)
