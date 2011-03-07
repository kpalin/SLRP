#!/usr/bin/env python

from distutils.core import setup
import subprocess as sp

try:
    git_proc = sp.Popen(["git","describe"],stdout=sp.PIPE)
    git_version = git_proc.stdout.read().strip()
    git_proc.wait()
    if git_proc.returncode == 0:
        print "Successfull git call"
        open("SLRPlib/__version__.py","w").write("_version=\"%s\""%(git_version))
except OSError:
    pass
                            

from SLRPlib.__version__ import _version




setup(name='SLRP',
      version=_version,
      description='Systematic Long Range Phasing',
      author='Kimmo Palin',
      author_email='kimmo.palin@sanger.ac.uk',
      url='https://github.com/kpalin/SLRP',
      packages=['SLRPlib'],
      scripts=['scripts/SLRP'],
      license="GPLv2 or greater",
     )
