#!/usr/bin/env python

from distutils.core import setup
import subprocess as sp

try:
    git_proc = sp.Popen(["git","describe"],stdout=sp.PIPE)
    git_version = git_proc.stdout.read().strip()
    git_proc.wait()
    if git_proc.returncode == 0:
        print "Successfull git call"
        open("SLRPlib/version.py","w").write("__version__=\"%s\""%(git_version))
except OSError:
    pass
                            

import SLRPlib




setup(name='SLRP',
      version = SLRPlib.__version__,
      description='Systematic Long Range Phasing',
      author='Kimmo Palin',
      author_email='kimmo.palin@sanger.ac.uk',
      url='https://github.com/kpalin/SLRP',
      packages=['SLRPlib',"SLRPlib/test"],
      scripts=['scripts/SLRP'],
      license="GPLv2 or greater",
     )
