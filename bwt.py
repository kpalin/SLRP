import itertools as its

def ilen(it):
    '''Return the length of an iterable.
    
    >>> ilen(range(7))
    7
    '''
    return sum(1 for _ in it)

def runlength_enc(xs):
    '''Return a run-length encoded version of the stream, xs.
    
    The resulting stream consists of (count, x) pairs.
    
    >>> ys = runlength_enc('AAABBCCC')
    >>> next(ys)
    (3, 'A')
    >>> list(ys)
    [(2, 'B'), (3, 'C')]
    '''
    return ((ilen(gp), x) for x, gp in its.groupby(xs))

def runlength_dec(xs):
    '''Expand a run-length encoded stream.
    
    Each element of xs is a pair, (count, x).
    
    >>> ys = runlength_dec(((3, 'A'), (2, 'B')))
    >>> next(ys)
    'A'
    >>> ''.join(ys)
    'AABB'
    '''
    return its.chain(*(its.repeat(x, n) for n, x in xs))
#    return its.chain.from_iterable(its.repeat(x, n) for n, x in xs)


def bwt(s):
    s = s + '\0'
    return ''.join([x[-1] for x in
                    sorted([s[i:] + s[:i] for i in range(len(s))])])

def ibwt(s):
    L = [''] * len(s)
    for i in range(len(s)):
        L = sorted([s[i] + L[i] for i in range(len(s))])
    return [x for x in L if x.endswith('\0')][0][:-1]


