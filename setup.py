#!/usr/bin/env python

from distutils.core import setup
import subprocess as sp
import os

try:
    git_proc = sp.Popen(["git","log","--pretty=format:'%h'","-n","1"],stdout=sp.PIPE,stderr=open(os.devnull,"w"))
    git_version = git_proc.stdout.read().strip()
    git_proc.wait()
    if git_proc.returncode == 0:
        print "Successfull git call"
        open("SLRPlib/version.py","w").write("__version__=\"%s\""%(git_version))
except OSError:
    raise
#pass
                            

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

# This will fail if you can't build c_scan_ext.so
import SLRPlib.scanTools

