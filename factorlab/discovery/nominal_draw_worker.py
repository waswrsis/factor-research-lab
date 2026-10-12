"""Fixed ten-second systemd entry; exclusive marker before project imports.

All imports run after systemd ExecMainStart and count against RuntimeMaxSec=10.
No lease/native metadata is sampled here; original draw issues the sole lease.
"""
import hashlib
import os
from pathlib import Path
import sys

def claim_paths(directory,cell,index):
    root=Path(os.environ.get('FACTOR_RESEARCH_ROOT',''))
    path=Path(directory)
    if (not root.is_absolute() or root.resolve()!=root or not path.is_absolute()
        or path.resolve()!=path or not path.is_relative_to(root) or type(cell)is not str
        or not cell or '\x00' in cell or len(cell)>256 or type(index)is not int or index<0):
        raise ValueError('Fixed original canonical registry coordinate required')
    key=hashlib.sha256((cell+'\x00'+str(index)).encode('utf8')).hexdigest()
    return (path/'native-draw/start-claims'/key,path/'native-draw-start-claims'/key)

def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if len(args)!=6 or args[0]!='--registry' or args[2]!='--cell' or args[4]!='--index':
        raise ValueError('Fixed registry/cell/index CLI only')
    index=int(args[5])
    if str(index)!=args[5] or index<0:raise ValueError('Canonical nonnegative index required')
    if sys.platform!='linux':raise OSError('Actual Linux worker required')
    for path in claim_paths(args[1],args[3],index):
        if path.parent.resolve()!=path.parent or not path.parent.is_dir():raise ValueError('Original pre-registered claim directory required')
        with path.open('xb')as stream:
            stream.write(b'original-draw-started-once\n');stream.flush();os.fsync(stream.fileno())
    # Imported only after the permanent claim. No process/boot/cgroup query or
    # lease issue precedes the original allocation in nominal_actual_trial.
    from .nominal_draw_native import run_worker
    return run_worker(args[1],args[3],index)

if __name__=='__main__':raise SystemExit(main())
