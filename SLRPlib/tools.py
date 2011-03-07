
def addSuffix(base,suffix):
   "Append a suffix to the base such that .gz ending is kept last"
   if base.endswith(".gz"):
      out = base[:-3] + suffix.strip() + ".gz"
   else:
      out = base + suffix

   return out


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

