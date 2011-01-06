# Speed improvement over numpy version.
import numpy as np
import cStringIO
import os
import itertools

from cPickle import load as _cload, loads
from numpy.lib._datasource import DataSource
from numpy.lib._compiled_base import packbits, unpackbits

from numpy.lib._iotools import LineSplitter, NameValidator, StringConverter, \
                     _is_string_like, has_nested_fields, flatten_dtype

from numpy.lib.io import _getconv

_file = file
_string_like = _is_string_like

def loadtxt(fname, dtype=float, comments='#', delimiter=None, converters=None,
            skiprows=0, usecols=None, unpack=False,count = -1):
    """
    Load data from a text file.

    Each row in the text file must have the same number of values.

    Parameters
    ----------
    fname : file or string
        File or filename to read.  If the filename extension is ``.gz`` or
        ``.bz2``, the file is first decompressed.
    dtype : data-type
        Data type of the resulting array.  If this is a record data-type,
        the resulting array will be 1-dimensional, and each row will be
        interpreted as an element of the array.   In this case, the number
        of columns used must match the number of fields in the data-type.
    comments : string, optional
        The character used to indicate the start of a comment.
    delimiter : string, optional
        The string used to separate values.  By default, this is any
        whitespace.
    converters : {}
        A dictionary mapping column number to a function that will convert
        that column to a float.  E.g., if column 0 is a date string:
        ``converters = {0: datestr2num}``. Converters can also be used to
        provide a default value for missing data:
        ``converters = {3: lambda s: float(s or 0)}``.
    skiprows : int
        Skip the first `skiprows` lines.
    usecols : sequence
        Which columns to read, with 0 being the first.  For example,
        ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
    unpack : bool
        If True, the returned array is transposed, so that arguments may be
        unpacked using ``x, y, z = loadtxt(...)``

    Returns
    -------
    out : ndarray
        Data read from the text file.

    See Also
    --------
    scipy.io.loadmat : reads Matlab(R) data files

    Examples
    --------
    >>> from StringIO import StringIO   # StringIO behaves like a file object
    >>> c = StringIO("0 1\\n2 3")
    >>> np.loadtxt(c)
    array([[ 0.,  1.],
           [ 2.,  3.]])

    >>> d = StringIO("M 21 72\\nF 35 58")
    >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
    ...                      'formats': ('S1', 'i4', 'f4')})
    array([('M', 21, 72.0), ('F', 35, 58.0)],
          dtype=[('gender', '|S1'), ('age', '<i4'), ('weight', '<f4')])

    >>> c = StringIO("1,0,2\\n3,0,4")
    >>> x,y = np.loadtxt(c, delimiter=',', usecols=(0,2), unpack=True)
    >>> x
    array([ 1.,  3.])
    >>> y
    array([ 2.,  4.])

    """
    user_converters = converters

    if usecols is not None:
        usecols = list(usecols)

    isstring = False
    if _is_string_like(fname):
        isstring = True
        if fname.endswith('.gz'):
            import gzip
            fh = seek_gzip_factory(fname)
        elif fname.endswith('.bz2'):
            import bz2
            fh = bz2.BZ2File(fname)
        else:
            fh = file(fname)
    elif hasattr(fname, 'readline'):
        fh = fname
    else:
        raise ValueError('fname must be a string or file handle')
    X = []

    def flatten_dtype(dt):
        """Unpack a structured data-type."""
        if dt.names is None:
            return [dt]
        else:
            types = []
            for field in dt.names:
                tp, bytes = dt.fields[field]
                flat_dt = flatten_dtype(tp)
                types.extend(flat_dt)
            return types

    def split_line(line):
        """Chop off comments, strip, and split at delimiter."""
        line = line.split(comments)[0].strip()
        if line:
            return line.split(delimiter)
        else:
            return []

    try:
        # Make sure we're dealing with a proper dtype
        dtype = np.dtype(dtype)
        defconv = _getconv(dtype)

        # Skip the first `skiprows` lines
        for i in xrange(skiprows):
            fh.readline()

        # Read until we find a line with some values, and use
        # it to estimate the number of columns, N.
        first_vals = None
        while not first_vals:
            first_line = fh.readline()
            if first_line == '': # EOF reached
                raise IOError('End-of-file reached before encountering data.')
            first_vals = split_line(first_line)
        N = len(usecols or first_vals)

        dtype_types = flatten_dtype(dtype)
        if len(dtype_types) > 1:
            # We're dealing with a structured array, each field of
            # the dtype matches a column
            converters = [_getconv(dt) for dt in dtype_types]
        else:
            # All fields have the same dtype
            converters = [defconv for i in xrange(N)]

        # By preference, use the converters specified by the user
        for i, conv in (user_converters or {}).iteritems():
            if usecols:
                try:
                    i = usecols.index(i)
                except ValueError:
                    # Unused converter specified
                    continue
            converters[i] = conv

        # Parse each line, including the first
        vals_gen = ( split_line(line) for line in itertools.chain([first_line], fh) )
        if usecols:
            vals_gen = ( [x for i,x in enumerate(vals) if i in usecols] for vals in vals_gen )
        data_gen = ( tuple([conv(val) for (conv, val) in zip(converters, vals)]) for vals in vals_gen if len(vals)>0 )

        if len(dtype_types) > 1:
            X = np.fromiter(data_gen, dtype = np.dtype([('', t) for t in dtype_types]),count=count)
            X = X.view(dtype)
        else:
            X = np.fromiter(itertools.chain(*data_gen), dtype = dtype,count=count)
    finally:
        if isstring:
            fh.close()


    X = np.squeeze(X)
    if unpack:
        return X.T
    else:
        return X
