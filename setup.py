#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pkg_resources import parse_version
import os, sys

numpy_min_version = '1.8'

def get_numpy_status():

    """
    Returns a dictionary containing a boolean specifying whether NumPy
    is up-to-date, along with the version string (empty string if
    not installed).
    """

    numpy_status = {}
    try:
        import numpy
        numpy_version = numpy.__version__
        numpy_status['up_to_date'] = parse_version(
            numpy_version) >= parse_version(numpy_min_version)
        numpy_status['version'] = numpy_version
    except ImportError:
        numpy_status['up_to_date'] = False
        numpy_status['version'] = ""
    return numpy_status

def setup_miccs():
    numpy_status = get_numpy_status()
    numpy_req_str = "MICCS requires NumPy >= {0}.\n".format(numpy_min_version)      

    if numpy_status['up_to_date'] is False:
        if numpy_status['version']:
            raise ImportError("Your installation of NumPy"
                              "{0} is out-of-date.\n{1}"
                              .format(numpy_status['version'], numpy_req_str))
        else:
            raise ImportError("NumPy is not installed.\n{0}"
                              .format(numpy_req_str))   

    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import setup, Extension
    from numpy.distutils.system_info import get_info

    lapack_info = get_info('lapack_opt',1)
    sources = [ 'miccs/miccs_core.f90',]

    if lapack_info:
        ext = Extension(name='miccs_core', sources=sources,
                        **lapack_info)

    if not lapack_info:
        raise('No BLAS or Lapack libraries found')

    setup(
        name='miccs',
        version="0.0.1",
        description="MICCS: Multiset Inverse Canonical Correlation Sparsification",
        author="Heejong Bong",
        author_email="hbong@andrew.cmu.edu",
        url="http://github.com/HeejongBong/miccs",
        license="Academic Free License",
        classifiers=[
            'License :: OSI Approved :: Academic Free License (AFL)',
            'Programming Language :: Python :: 3',
            'Programming Language :: Fortran',
            'Topic :: Scientific/Engineering',
            ],
        packages = ['miccs'],
        ext_modules = [ext])

if __name__ == '__main__':
    setup_miccs()