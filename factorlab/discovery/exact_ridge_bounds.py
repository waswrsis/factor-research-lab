"""Bounded exact-real certificates for given binary64 values, never a fit.

These certificates do NOT establish how a training matrix was formed, correct
rounding of production BLAS/libm, complete support, or a future cold learner.
All fractions use lossless hexadecimal integer numerator/denominator records.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import math
import os
import sys

MAX_ROWS=8192
MAX_COLUMNS=32
MAX_CELLS=262144
MAX_STEPS=1100000
MAX_BITS=32768
MAX_BOX_BITS=4096
SQRT_BITS=128
MAX_MODEL_BYTES=65536
MAX_RESULT_BYTES=2097152
EXACT_AFFINE_CONTRACT='exact-rational-stored-ridge-affine-v1'
SOURCE_FILES={
    'factorlab/discovery/convex_eval.py':'e61680e7c8ddd8ce85d4ef64f9e31b2c3ad2fc7f7e307300a5b1c1a30d816159',
    'factorlab/protocol.py':'12f8c10533292eb6a2f905ef62e212b2ddb94b9e22b5f2b4d485232d3952cf60',
}


class BoundError(ValueError):
    pass


class UnsupportedArithmetic(BoundError):
    pass


def _need(value,message):
    if not value:raise BoundError(message)


def _binary64(value):
    # Do not silently round Decimal, arbitrary int, numpy scalar or bool inputs.
    _need(type(value) is float and math.isfinite(value),'Finite builtin binary64 float required')
    _need(sys.float_info.radix==2 and sys.float_info.mant_dig==53 and sys.float_info.max_exp==1024,
          'Host float is not IEEE binary64')
    return Fraction.from_float(value)


def _sequence(value,n=None):
    _need(type(value) in (list,tuple),'Plain list or tuple required')
    if n is not None:_need(len(value)==n,'Shape mismatch')
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _bound_project_root():
    """Use actual pinned production protocol ROOT, never a caller root proof."""
    from factorlab import protocol
    module_path=Path(protocol.__file__)
    origin=getattr(getattr(protocol,'__spec__',None),'origin',None)
    _need(module_path.is_absolute() and module_path.resolve()==module_path and not module_path.is_symlink()
          and module_path.is_file() and module_path.stat().st_size<=1048576,
          'Canonical actual protocol module required')
    _need(origin is not None and Path(origin)==module_path and
          _file_hash(module_path)==SOURCE_FILES['factorlab/protocol.py'],
          'Loaded production protocol source identity differs')
    expected=Path('D:/FactorResearch') if os.name=='nt' else Path(
        os.environ.get('FACTOR_RESEARCH_ROOT',str(module_path.parent.parent)))
    root=protocol.ROOT
    _need(isinstance(root,Path) and root==expected and root.is_absolute() and root.resolve()==root
          and root.is_dir() and module_path==root/'factorlab/protocol.py',
          'Actual protocol ROOT/environment/loaded workspace differ')
    for name in ('fingerprint','project_path'):
        fn=getattr(protocol,name);code=getattr(fn,'__code__',None)
        _need(code is not None and Path(code.co_filename)==module_path and fn.__module__==protocol.__name__
              and fn.__globals__ is protocol.__dict__,'Loaded protocol callable identity differs')
    _need(protocol.project_path('.')==root,'Actual protocol project path differs')
    return root


ROOT=_bound_project_root()


def rational(value):
    """Lossless output format; both integers are retained, including denominator."""
    _need(type(value) is Fraction,'Fraction required')
    return {'numerator_hex':hex(value.numerator),'denominator_hex':hex(value.denominator)}


class _Ops:
    def __init__(self):self.steps=0
    def checked(self,value):
        _need(type(value) is Fraction and max(abs(value.numerator).bit_length(),value.denominator.bit_length())<=MAX_BITS,
              'Exact rational size limit exceeded')
        return value
    def tick(self,*values):
        self.steps+=1;_need(self.steps<=MAX_STEPS,'Exact arithmetic work limit exceeded')
        for v in values:self.checked(v)
    def add(self,a,b):
        self.tick(a,b);return self.checked(a+b)
    def sub(self,a,b):
        self.tick(a,b);return self.checked(a-b)
    def mul(self,a,b):
        self.tick(a,b);return self.checked(a*b)
    def div(self,a,b):
        self.tick(a,b);_need(b!=0,'Zero rational denominator');return self.checked(a/b)
    def square(self,a):return self.mul(a,a)
    def sum(self,values):
        out=Fraction(0)
        for value in values:out=self.add(out,value)
        return out
    def sqrt(self,value):
        """Floor/ceiling sqrt on a fixed 2^-128 grid, using integer isqrt only."""
        self.tick(value);_need(value>=0,'Negative square root')
        scaled,remainder=divmod(value.numerator << (2*SQRT_BITS),value.denominator)
        root=math.isqrt(scaled);exact=(remainder==0 and root*root==scaled)
        lower=Fraction(root,1<<SQRT_BITS)
        upper=lower if exact else Fraction(root+1,1<<SQRT_BITS)
        self.checked(lower);self.checked(upper)
        _need(self.square(lower)<=value<=self.square(upper),'Internal outward-square-root invariant')
        return lower,upper


def sqrt_bounds(value):
    """Public tiny rational root primitive; rejects non-Fraction and oversized input."""
    ops=_Ops();lower,upper=ops.sqrt(value)
    return {'radicand':rational(value),'lower':rational(lower),'upper':rational(upper),
            'grid_denominator_power_of_two':SQRT_BITS,'method':'integer-isqrt-exact-square-comparison'}


def _interval(lower,upper):return {'lower':rational(lower),'upper':rational(upper)}


def _finish(value):
    value['kernel_sha256']=_file_hash(Path(__file__).resolve())
    encoded=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    _need(len(encoded)<=MAX_RESULT_BYTES,'Certificate output size limit exceeded')
    return value


def certify_augmented_ridge(A_binary64,t_binary64,rho_binary64,beta_binary64):
    """Certify the supplied actual beta for [A; rho I], [t; 0] over exact reals.

    g=A^T(A beta-t)+rho^2 beta is the normal residual (half the objective
    gradient). No solver is imported or called. rho is the stored float itself.
    Dimensions <=8192 x32; caller must export ndarray float64 values with tolist.
    """
    _sequence(A_binary64);_sequence(beta_binary64)
    n,p=len(A_binary64),len(beta_binary64)
    _need(1<=n<=MAX_ROWS and 1<=p<=MAX_COLUMNS and n*p<=MAX_CELLS,'Matrix dimension/work limit')
    _sequence(t_binary64,n)
    for row in A_binary64:_sequence(row,p)
    ops=_Ops()
    A=[[_binary64(v) for v in row] for row in A_binary64]
    t=[_binary64(v) for v in t_binary64];beta=[_binary64(v) for v in beta_binary64]
    rho=_binary64(rho_binary64);_need(rho>0,'rho must be strictly positive')
    lam=ops.square(rho)
    residual=[ops.sub(ops.sum(ops.mul(a,b) for a,b in zip(row,beta)),y) for row,y in zip(A,t)]
    beta2=ops.sum(ops.square(v) for v in beta)
    residual2=ops.sum(ops.square(v) for v in residual)
    objective=ops.add(residual2,ops.mul(lam,beta2))
    g=[ops.add(ops.sum(ops.mul(A[i][j],residual[i]) for i in range(n)),ops.mul(lam,beta[j])) for j in range(p)]
    g2=ops.sum(ops.square(v) for v in g);t2=ops.sum(ops.square(v) for v in t)
    beta_lo,beta_hi=ops.sqrt(beta2)
    j_lo,j_hi=ops.sqrt(ops.div(objective,lam))
    t_lo,t_hi=ops.sqrt(t2);g_lo,g_hi=ops.sqrt(g2)
    twice_rho=ops.mul(Fraction(2),rho)
    residual_lo=ops.add(ops.div(t_lo,twice_rho),ops.div(g_lo,lam))
    residual_hi=ops.add(ops.div(t_hi,twice_rho),ops.div(g_hi,lam))
    optimum_distance_hi=ops.div(g_hi,lam)
    # Exact comparisons also validate these upper bounds for this actual beta.
    for upper in (beta_hi,j_hi,residual_hi):_need(beta2<=ops.square(upper),'Computed upper bound failed exact squared check')
    tight_hi=min(beta_hi,j_hi,residual_hi)
    commitment={'A':[[v.hex() for v in row] for row in A_binary64],
                't':[v.hex() for v in t_binary64],'rho':rho_binary64.hex(),'beta':[v.hex() for v in beta_binary64]}
    return _finish({'schema':'exact-augmented-ridge-certificate-v1','input_sha256':_hash(commitment),
        'shape':[n,p],'input_semantics':'exact-real-values-of-supplied-binary64-elements',
        'rho':rational(rho),'lambda_effective':rational(lam),
        'objective':rational(objective),'data_residual_squared':rational(residual2),
        'normal_residual':[rational(v) for v in g],'normal_residual_squared':rational(g2),
        'target_norm_squared':rational(t2),'actual_coefficient_norm_squared':rational(beta2),
        'actual_coefficient_norm':_interval(beta_lo,beta_hi),
        'objective_coefficient_bound':_interval(j_lo,j_hi),
        'normal_residual_coefficient_bound':_interval(residual_lo,residual_hi),
        'actual_coefficient_upper':rational(tight_hi),
        'distance_to_exact_optimizer_upper':rational(optimum_distance_hi),
        'normal_residual_definition':'A^T(A beta-t)+rho^2 beta; NOT full 2g gradient',
        'bound_theorems':['||beta||^2 <= J/rho^2',
          '||beta-beta_star|| <= ||g||/rho^2',
          '||beta_star|| <= ||t||/(2 rho)'],
        'scope':'single-supplied-frozen-matrix-and-actual-coefficient-vector',
        'matrix_formation_and_fit_provenance':'NOT_ESTABLISHED',
        'production_binary64_rounding':'NOT_PROVED',
        'cold_learner_or_full_support':'NOT_ESTABLISHED','arithmetic_steps':ops.steps})


def _unique(pairs):
    value={}
    for key,item in pairs:
        _need(key not in value,'Duplicate JSON key');value[key]=item
    return value


def _validate_json_tree(value,depth=0,count=None):
    if count is None:count=[0]
    count[0]+=1;_need(depth<=16 and count[0]<=4096,'Saved JSON structural limit')
    if type(value) is dict:
        for key,item in value.items():
            _need(type(key) is str,'JSON object key type');_validate_json_tree(item,depth+1,count)
    elif type(value) is list:
        for item in value:_validate_json_tree(item,depth+1,count)
    elif type(value) is float:_need(math.isfinite(value),'Nonfinite JSON value')
    else:_need(value is None or type(value) in (str,int,bool),'Unsupported JSON scalar')


def _read_saved_model(path,expected_sha256):
    _need(type(path) in (str,Path) or isinstance(path,Path),'Saved model path required')
    path=Path(path)
    _need(path.is_absolute() and path.resolve()==path and not path.is_symlink() and path.is_file(),
          'Canonical existing saved model file required')
    _need(type(expected_sha256) is str and len(expected_sha256)==64 and all(c in '0123456789abcdef' for c in expected_sha256),
          'Explicit saved file SHA256 required')
    _need(path.stat().st_size<=MAX_MODEL_BYTES,'Saved model byte limit')
    raw=path.read_bytes();_need(len(raw)<=MAX_MODEL_BYTES,'Saved model grew beyond limit')
    _need(hashlib.sha256(raw).hexdigest()==expected_sha256,'Saved model file identity mismatch')
    try:payload=json.loads(raw,object_pairs_hook=_unique)
    except (RecursionError,UnicodeDecodeError,ValueError) as exc:raise BoundError('Invalid bounded model JSON') from exc
    _validate_json_tree(payload)
    _need(type(payload) is dict and set(payload)=={'digest','model'},'Exact saved Ridge payload required')
    model=payload['model']
    _need(type(model) is dict and set(model)=={'kind','version','scaler','coefficients','intercept','ridge',
          'feature_names','input_schema_hash','provenance','target_floor'},'Unknown saved model schema')
    _need(type(payload['digest']) is str and _hash(model)==payload['digest'],'Saved model digest mismatch')
    _need(model['kind']=='v7_log_ridge' and type(model['version']) is int and model['version']==1,'Unsupported model kind/version')
    _need(type(model['target_floor']) is float and model['target_floor']==1e-8,'Unknown target/decode floor')
    names=_sequence(model['feature_names']);p=len(names)
    _need(1<=p<=MAX_COLUMNS and all(type(n) is str and 0<len(n.encode('utf8'))<=256 for n in names),'Feature name/count limit')
    _need(len(set(names))==p,'Duplicate feature names')
    schema=model['input_schema_hash']
    _need(schema is None or (type(schema) is str and len(schema)==64 and all(c in '0123456789abcdef' for c in schema)),
          'Unknown feature schema digest')
    _need(type(model['provenance']) is dict,'Ridge provenance must be object')
    beta=[_binary64(v) for v in _sequence(model['coefficients'],p)]
    intercept=_binary64(model['intercept']);_need(_binary64(model['ridge'])>0,'Positive actual ridge required')
    scaler=model['scaler'];_need(type(scaler) is dict and set(scaler)=={'center','scale','bypass_mask'},'Unknown saved scaler schema')
    center=[_binary64(v) for v in _sequence(scaler['center'],p)]
    scale=[_binary64(v) for v in _sequence(scaler['scale'],p)]
    bypass=_sequence(scaler['bypass_mask'],p)
    _need(all(type(b) is bool for b in bypass),'Bypass mask requires exact bool values')
    _need(all(s>0 for s in scale),'Saved scale must be strictly positive; no silent floor')
    _need(all(not b or (c==0 and s==1) for b,c,s in zip(bypass,center,scale)),
          'Bypassed training scaler must store center 0 and scale 1')
    root=_bound_project_root()
    _need(root==ROOT,'Bound production workspace changed since kernel import')
    for rel,digest in SOURCE_FILES.items():
        source=root/rel
        _need(source.resolve()==source and not source.is_symlink() and source.is_file() and source.stat().st_size<=1048576,
              'Original model source path/size changed')
        _need(_file_hash(source)==digest,'Original saved Ridge source semantics changed')
    return path,expected_sha256,payload,center,scale,beta,intercept


def _box_scalar(value):
    if type(value) is float:return _binary64(value)
    _need(type(value) is Fraction,'Box endpoint must be binary64 float or exact Fraction')
    _need(max(abs(value.numerator).bit_length(),value.denominator.bit_length())<=MAX_BOX_BITS,'Input box rational bit limit')
    return value


def bound_frozen_predictor(saved_model_path,raw_feature_box,arithmetic_contract,*,expected_model_sha256,requested_output='log_prediction'):
    """Exact rational image of an actual saved Ridge scaler and affine map.

    raw_feature_box contains exact feature_names, input_schema_hash, bounds.
    A box is only a supplied set, NOT a claim about observed/full data support.
    The sole accepted contract declares exact-real interpretation; callers
    cannot provide a bool or a JSON 'proof' to certify production rounding.
    """
    if type(arithmetic_contract) is not str or arithmetic_contract!=EXACT_AFFINE_CONTRACT:
        raise UnsupportedArithmetic('Known exact-rational affine contract required; production arithmetic unproved')
    if requested_output!='log_prediction':
        raise UnsupportedArithmetic('Decoder/loss require a separately implemented, source-bound arithmetic contract')
    path,digest,payload,center,scale,beta,intercept=_read_saved_model(saved_model_path,expected_model_sha256)
    model=payload['model'];p=len(beta)
    _need(type(raw_feature_box) is dict and set(raw_feature_box)=={'feature_names','input_schema_hash','bounds'},'Exact feature box schema required')
    _need(raw_feature_box['feature_names']==model['feature_names'] and raw_feature_box['input_schema_hash']==model['input_schema_hash'],
          'Feature names/order or feature-policy binding mismatch')
    _sequence(raw_feature_box['bounds'],p);bounds=[];ops=_Ops()
    for row in raw_feature_box['bounds']:
        _sequence(row,2);lo,hi=map(_box_scalar,row);_need(lo<=hi,'Reversed feature interval');bounds.append((lo,hi))
    scaled=[(ops.div(ops.sub(lo,c),s),ops.div(ops.sub(hi,c),s)) for (lo,hi),c,s in zip(bounds,center,scale)]
    lo=hi=intercept;terms=[]
    for (a,b),coefficient in zip(scaled,beta):
        x,y=ops.mul(a,coefficient),ops.mul(b,coefficient);term=(min(x,y),max(x,y));terms.append(term)
        lo=ops.add(lo,term[0]);hi=ops.add(hi,term[1])
    largest=_binary64(sys.float_info.max)
    exceeded=any(abs(v)>largest for pair in [*scaled,*terms,(lo,hi)] for v in pair)
    # Recheck the original model bytes after the bounded calculation.
    _need(_file_hash(path)==digest,'Saved model changed during exact propagation')
    box_identity={'feature_names':model['feature_names'],'input_schema_hash':model['input_schema_hash'],
                  'bounds':[_interval(*pair) for pair in bounds]}
    return _finish({'schema':'exact-frozen-ridge-affine-box-v1','model_file_sha256':digest,'model_digest':payload['digest'],
        'model_source_files':dict(SOURCE_FILES),'feature_box_sha256':_hash(box_identity),
        'raw_feature_box':box_identity,'scaled_feature_box':[_interval(*pair) for pair in scaled],
        'affine_log_prediction':_interval(lo,hi),'arithmetic_contract':EXACT_AFFINE_CONTRACT,
        'arithmetic_scope':'exact-real-scaler-and-affine-map-of-stored-binary64-parameters',
        'production_binary64_conformance':'NOT_PROVED',
        'finite_binary64_range_diagnostic':'EXACT_ENDPOINT_OUTSIDE_BINARY64_FINITE_RANGE' if exceeded else 'ENDPOINTS_WITHIN_RANGE_ONLY_NOT_ROUNDING_PROOF',
        'decoder':{'status':'UNSUPPORTED_NO_SOURCE_BOUND_LIBM_ROUNDING_CONTRACT','interval':None},
        'loss':{'status':'UNSUPPORTED_DECODER_AND_TARGET_ARITHMETIC','interval':None},
        'scope':'single-saved-model-and-supplied-box-only',
        'raw_box_support_coverage':'NOT_ESTABLISHED','matrix_fit_link':'NOT_ESTABLISHED',
        'cold_learner_or_full_support':'NOT_ESTABLISHED','arithmetic_steps':ops.steps})
