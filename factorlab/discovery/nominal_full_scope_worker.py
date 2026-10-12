"""Fixed original full-policy worker; retained natural exit confers OS proof."""
import os
from pathlib import PurePosixPath
import re
import sys
import threading

def _arguments(argv):
    if (type(argv)is not list or len(argv)!=8 or any(type(v)is not str for v in argv)
        or argv[::2]!=['--registry','--cell','--trial-index','--oracle-index']):
        raise ValueError('Fixed full-scope registry/cell/trial/oracle arguments required')
    directory,cell,index,oracle=argv[1::2]
    if (not 0<len(directory)<=4096 or not directory.startswith('/')or directory.startswith('//')
        or str(PurePosixPath(directory))!=directory or '..'in PurePosixPath(directory).parts
        or any(ord(c)<32 or ord(c)==127 for c in directory)
        or re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]{0,79}',cell)is None
        or re.fullmatch(r'0|[1-9][0-9]{0,3}',index)is None
        or (oracle!='validation'and re.fullmatch(r'0|[1-9][0-9]{0,7}',oracle)is None)):
        raise ValueError('Canonical literal original coordinate required')
    return directory,cell,int(index),None if oracle=='validation'else int(oracle)

def main(argv=None):
    try:
        args=_arguments(sys.argv[1:]if argv is None else argv)
        if sys.platform!='linux'or os.name!='posix'or threading.current_thread()is not threading.main_thread():
            raise OSError('Actual Linux main-thread service required')
        from .nominal_full_scope_proof import run_coordinate,VERSION
        value=run_coordinate(*args)
        if (value['version']!=VERSION or value['provisional_only']is not True
            or value['native_completed']is not False or value['nominal_truth_certified']is not False):
            raise ValueError('Original provisional proof result unavailable')
        return 0
    except BaseException as error:
        print(type(error).__name__+': '+str(error)[:1000],file=sys.stderr);return 1

if __name__=='__main__':raise SystemExit(main())
