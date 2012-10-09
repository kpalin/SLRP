try:
    from version import __version__
except ImportError:
    # In case we haven't correctly created the version file (e.g. missing git)
    __version__="1.0-nogit"
