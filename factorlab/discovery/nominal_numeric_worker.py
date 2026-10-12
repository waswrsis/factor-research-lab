"""Fixed source entry. No launch, retry, alternate callable or new allowance."""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta
import sys

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--cutoff',required=True)
    args=parser.parse_args(argv)
    when=datetime.fromisoformat(args.cutoff)
    if (when.isoformat()!=args.cutoff or when.utcoffset()!=timedelta(0)
        or when.day!=1 or (when.hour,when.minute,when.second,when.microsecond)!=(0,0,0,0)):
        raise ValueError('Original UTC month boundary required')
    if sys.platform!='linux':raise OSError('Original Linux source-bound fixed pair worker required')
    from .nominal_numeric_history import run_original_pair,VERSION
    result=run_original_pair(args.protocol,args.cutoff)
    if (type(result)is not dict or set(result)!={'version','status','protocol_digest','cutoff','completion_hash','provisional_only','cleanup_signals'}
        or result['version']!=VERSION or result['status']not in ('numeric_failed_consumed','models_completed')
        or result['cutoff']!=args.cutoff or result['provisional_only']is not True or result['cleanup_signals']is not False
        or any(type(result[k])is not str or len(result[k])!=64 or any(c not in '0123456789abcdef'for c in result[k])for k in ('protocol_digest','completion_hash'))):
        raise ValueError('Original completed provisional worker result required')
    return 0

if __name__=='__main__':raise SystemExit(main())
