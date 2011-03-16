
def addSuffix(base,suffix):
   "Append a suffix to the base such that .gz ending is kept last"
   if base.endswith(".gz"):
      out = base[:-3] + suffix.strip() + ".gz"
   else:
      out = base + suffix

   return out


def uniqueRows(a):
   "Return array containing unique rows of a"
   import numpy
   b=numpy.unique(a.view('S%d'%(a.itemsize*a.shape[1]))).view(a.dtype)
   b=b.reshape((-1,a.shape[1]))
   return b
   
def printerr(*val):
   "Print to standard error. Might conflict with python3"
   import sys
   sys.stderr.write(" ".join(map(str, val))+"\n")


def setup_post_mortem():
    "Set up system to start debugger after raised exception"
    import pdb
    import sys
    def info(err_type, value, tb):
        import pdb
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(err_type, value, tb)
        else:
            import traceback
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(err_type, value, tb)
            print
            # ...then start the debugger in post-mortem mode.
            pdb.pm()

    sys.excepthook = info


def lineno():
    """Returns the current line number in our program.

    Danny Yoo (dyoo@hkn.eecs.berkeley.edu)
    """
    import inspect
    return inspect.currentframe().f_back.f_lineno
    

def sortInplaceByCol(a,col=0):
    "Sort a normal 2d numpy array according to col column. First by default."
    recTypes = [("f%d"%(i),a.dtype) for i in range(a.shape[1])]
    v = a.view(recTypes)
    v.sort(order=[ recTypes[col][0] ],axis=0)
    return v


def sortDescendingByCol(a,col=0):
    "Sort a normal 2d numpy array to descending order according to col column. First by default. Not in place"
    v = a[:,col].argsort()[::-1]
    return a[v]
    
def _functionId(nFramesUp=0):
    """ Create a string naming the function n frames up on the stack.
    """
    import sys
    co = sys._getframe(nFramesUp+1).f_code
    return "%s (%s @ %d)" % (co.co_name, co.co_filename, co.co_firstlineno)

