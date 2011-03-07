#!/usr/bin/env python

from distutils.core import setup
import subprocess as sp

setup(name='SLRP',
      version="$Id$,
      description='Systematic Long Range Phasing',
      author='Kimmo Palin',
      author_email='kimmo.palin@sanger.ac.uk',
      url='https://github.com/kpalin/SLRP',
      packages=['SLRPlib'],
      scripts=['scripts/SLRP'],
      license="GPLv2 or greater",
     )
