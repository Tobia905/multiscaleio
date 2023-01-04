import os

class FitlogSuppressor(object):
    """
    A context manager for doing a "deep suppression" 
    of stdout and stderr in python. This is needed 
    since standard logging disabling is not working
    for Prophet fit infos. 
    """
    def __init__(self):
        # open null files
        self.null_log = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # aave the actual stdout (1) and stderr (2) 
        self.real_std = (os.dup(1), os.dup(2))

    def __enter__(self):
        # assign the null to stdout and stderr.
        os.dup2(self.null_log[0], 1)
        os.dup2(self.null_log[1], 2)

    def __exit__(self, *_):
        # re-assign the real stdout/stderr
        os.dup2(self.real_std[0], 1)
        os.dup2(self.real_std[1], 2)
        # close the null files
        os.close(self.null_log[0])
        os.close(self.null_log[1])