
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
