"""Private registered-nominal prefix session; no cache is an authorization flag.

This core does not alter old readers. The read-grant fallback still runs their
full snapshot reconstruction. It makes no end-to-end linearity/WCET claim and
cannot detect arbitrary disk edits instantly without rereading those bytes.
"""
from __future__ import annotations

import copy
import json
import os
import threading
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path

from factorlab.protocol import fingerprint, project_path
from .evidence import file_hash
from .monthly_work_authority import source_files
from .frame_identity import frame_identity

RELATIVE = 'factorlab/discovery/nominal_prefix_session.py'
_ACTIVE = ContextVar('original_nominal_prefix_session', default=None)
_KEY = object()


def _bytes(value):
    return json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode('utf8')


def _decode(value):
    return json.loads(value)


def _installed():
    files = source_files()
    if Path(__file__).resolve()!=project_path(RELATIVE).resolve() or files.get(RELATIVE)!=file_hash(__file__):
        raise ValueError('Canonical integrated nominal session source required')
    return files


def _db_identity(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Original database path required')
    s=path.stat()
    # Only replacement detection. These fields are NOT proof that bytes stayed unchanged.
    return (str(path.resolve()),s.st_dev,s.st_ino)


class _AppendGrant:
    def __init__(self,key,session,generation,tail=()):
        if key is not _KEY:raise TypeError('Only the original active session can issue an append grant')
        self._session,self._generation,self._used=session,generation,False
        self._before_tail=_bytes(tail)
        self._head=_decode(session._state)['head'];self._batch_digest=fingerprint(session._batch.binding)


class _ReadGrant:
    def __init__(self,key,session,prefix,frame):
        if key is not _KEY:raise TypeError('Only an original snapshot read can issue a read grant')
        self._session,self._generation,self._closed=session,session._generation,False
        self._prefix=_bytes(prefix)
        self._frame=frame.copy(deep=True);self._frame.attrs=copy.deepcopy(frame.attrs)
        self._identity=frame_identity(self._frame);self._attrs=_bytes(self._frame.attrs)

    def _check(self):
        s=self._session;s._check()
        epoch=getattr(s,'_epoch',None)
        retained=(type(epoch) is _ReadEpoch and not epoch.closed and epoch.grant is self)
        if self._closed or self._generation!=s._generation or not (s._read is self or retained):
            raise ValueError('Read grant is closed, stale, or belongs to another session')
        if frame_identity(self._frame)!=self._identity or _bytes(self._frame.attrs)!=self._attrs:
            s._poison('private snapshot mutation')
            raise ValueError('Private original read snapshot changed')

    def frame(self):
        self._check()
        result=self._frame.copy(deep=True);result.attrs=copy.deepcopy(self._frame.attrs)
        return result

    def validate(self,frame):
        self._check()
        if frame_identity(frame)!=self._identity or _bytes(frame.attrs)!=self._attrs:
            self._session._poison('consumer frame differs')
            raise ValueError('Consumer bytes differ from its original read grant')
        return _decode(self._prefix)


class _ReadEpoch:
    """Private, one-frame-bounded capsule for one fixed committed generation."""
    def __init__(self,key,session):
        if key is not _KEY:raise TypeError('Only an original session issues a read epoch')
        self.session=session;self.generation=session._generation
        self.closed=False;self.key=None;self.grant=None

    def close(self):
        self.closed=True
        if self.grant is not None:self.grant._closed=True
        self.grant=None;self.key=None


class NominalPrefixSession:
    """A cooperative single-writer invocation, never a persistent success cache."""
    def __init__(self,key,trial,runtime,batch):
        if key is not _KEY:raise TypeError('Use original registered open_original')
        from . import nominal_trial_controller as controller
        self._controller,self._cache_var=controller,controller._READ
        self._trial,self._runtime,self._batch=trial,runtime,batch
        self._owner=(os.getpid(),threading.get_ident())
        self._source=_bytes(_installed());self._closed=False;self._poisoned=None
        self._append=None;self._read=None;self._paid=None;self._generation=0;self._auditing=False
        self._epoch=None;self._building_snapshot=False
        self._database_ids={p:_db_identity(p) for p in (runtime.directory/'lifecycle.sqlite',runtime.feed.directory/'receipts.sqlite')}
        self._lock=runtime.lifecycle.controller_lock()

    @classmethod
    @contextmanager
    def open_original(cls,trial,runtime,batch):
        from . import nominal_trial_controller as controller
        from .v7_forward import ForwardRuntime
        trial=controller._original_trial(trial)
        if (type(runtime) is not ForwardRuntime or type(batch) is not controller._PhysicalBatch
                or controller._BATCH.get() is not batch or batch.trial.digest!=trial.digest
                or runtime.directory!=trial.paths['runtime'] or _ACTIVE.get() is not None):
            raise ValueError('Original single registered trial/runtime/physical batch required')
        if any((p/'prefix-session-failure.json').exists() for p in (batch.directory,batch.mirror)):
            raise ValueError('An interrupted/failed prefix session cannot be restarted')
        obj=cls(_KEY,trial,runtime,batch)
        with obj._lock:
            token=_ACTIVE.set(obj)
            try:
                obj._full_audit(install=True)
                yield obj
                if obj._append is not None or obj._read is not None or obj._paid is not None or obj._epoch is not None:
                    raise ValueError('Original invocation ended with an unfinished grant')
                obj._full_audit(install=False)
            except BaseException as error:
                obj._poison(type(error).__name__+':'+str(error)[:500]);raise
            finally:
                obj._closed=True;_ACTIVE.reset(token)

    def _poison(self,reason):
        if self._poisoned is None:
            self._poisoned=reason
            from .v7_storage import immutable_record
            value={'trial_digest':self._trial.digest,'batch':self._batch.binding,
                   'reason':reason,'consumed':True,'retain_N':True,'formal_acceptance':False}
            self._failure_write_errors=[]
            for path in (self._batch.directory,self._batch.mirror):
                try:immutable_record(path/'prefix-session-failure.json',value)
                except BaseException as error:self._failure_write_errors.append(type(error).__name__)
        if self._read is not None:self._read._closed=True
        if self._epoch is not None:self._epoch.close()

    def _check(self):
        from . import nominal_trial_controller as controller
        if (self._closed or self._poisoned is not None or _ACTIVE.get() is not self
                or self._owner!=(os.getpid(),threading.get_ident())
                or controller._BATCH.get() is not self._batch):
            raise ValueError('Original nominal session is absent, poisoned, or moved')
        self._batch.check()
        if any(_db_identity(p)!=identity for p,identity in self._database_ids.items()):
            self._poison('database replacement');raise ValueError('Original database replaced')

    def _full_audit(self,install=False):
        if self._auditing:raise ValueError('Recursive complete audit cannot borrow session state')
        self._auditing=True
        try:return self._audit_original(install)
        finally:self._auditing=False

    def _audit_original(self,install=False):
        from . import nominal_trial_controller as controller
        from .calibration_runtime_contract import _original_operations
        self._check()
        if controller._READ.get() is not None:
            raise ValueError('Full audit cannot borrow an older cached observation read')
        if _bytes(_installed())!=self._source:raise ValueError('Canonical source changed')
        obj=controller.from_lifecycle(self._runtime.lifecycle)
        if type(obj) is not controller.NominalObservation or obj.trial.digest!=self._trial.digest:
            raise ValueError('Original nominal observation enrollment required')
        # Begin with real uncached disk verification, not the retained old scope.
        # Existing full validators remain authoritative; this fallback is not O(H).
        self._runtime.feed.verify()
        state=obj.inspect(self._runtime.lifecycle);raw=obj.raw()
        self._check_receipt_names(obj,state['events'])
        operations=_original_operations(self._runtime.lifecycle)
        if not install and (_bytes(obj.identity)!=self._identity or _bytes(raw)!=self._raw
                            or _bytes(state)!=self._state):
            raise ValueError('Fresh audit epoch differs before any nested reader can borrow it')
        fresh_scope={'directory':str(self._runtime.directory),'state':copy.deepcopy(state)}
        fresh_bytes=_bytes(fresh_scope)
        # Existing model readers nest observation_read_scope. Reuse only THIS
        # freshly verified invocation's state while retaining the controller and
        # (when called from paid preparation) already-held original journal lock.
        # Setting None through nested readers would reacquire the same OS lock
        # through a different FileLock instance. No old-cache success is reused.
        fresh_token=controller._READ.set(fresh_scope)
        try:
            self._runtime.lifecycle.verify()
            if controller._READ.get() is not fresh_scope or _bytes(fresh_scope)!=fresh_bytes:
                raise ValueError('Fresh audit scope changed during original model/ledger verification')
        finally:controller._READ.reset(fresh_token)
        # After releasing only the transient cache, reread the actual disk again.
        if (obj.inspect(self._runtime.lifecycle)!=state or obj.raw()!=raw
                or _original_operations(self._runtime.lifecycle)!=operations
                or _bytes(_installed())!=self._source):
            raise ValueError('Original disk/source/LC changed during complete audit')
        advances=[o for o in operations if o['kind']==controller.ADVANCED]
        prefixes=tuple(_bytes(obj._prefix(hour,op,controller._read(obj.directory/'events'/f'{hour:05d}.json')['source']))
                       for hour,op in enumerate(advances))
        if install:
            self._observer=obj;self._identity=_bytes(obj.identity);self._raw=_bytes(raw)
            self._raw_private=_decode(self._raw);self._state=_bytes(state);self._prefixes=prefixes
            self._generation=state['events'];self._tail=_bytes(operations[-1]) if operations else None
            self._sequence=len(operations)
        elif (_bytes(obj.identity)!=self._identity or _bytes(raw)!=self._raw
                or _bytes(state)!=self._state or prefixes!=self._prefixes):
            raise ValueError('Complete original source/prefix differs from private validated snapshot')
        self._check()
        return state

    @staticmethod
    def _check_receipt_names(obj,count):
        expected={f'{hour:05d}-{suffix}.json' for hour in range(count) for suffix in ('intent','complete')}
        for path in (obj.trial.paths['root']/'generator_publications',obj.mirror/'receipts'):
            if {p.name for p in path.glob('*.json')}!=expected:
                raise ValueError('Partial or extra original publication allocation')

    def _read_tail(self):
        """Hash actual bytes read, not metadata or an earlier verification flag."""
        db=self._runtime.lifecycle.db
        if self._sequence:
            row=db.execute('SELECT record,hash FROM operations WHERE seq=?',(self._sequence,)).fetchone()
            if row is None or _bytes(json.loads(row[0]))!=self._tail or fingerprint(json.loads(row[0]))!=row[1]:
                raise ValueError('Stored previous LC head bytes changed')
        prior='0'*64 if self._tail is None else fingerprint(_decode(self._tail));sequence=self._sequence;tail=[]
        for seq,text,digest in db.execute('SELECT seq,record,hash FROM operations WHERE seq>? ORDER BY seq',(sequence,)):
            value=json.loads(text)
            if seq!=sequence+1 or value['seq']!=seq or value['previous']!=prior or fingerprint(value)!=digest:
                raise ValueError('Actual lifecycle suffix chain differs')
            sequence=seq;prior=digest;tail.append(value)
        return tail

    def begin_append(self):
        self._check()
        if self._append is not None or self._read is not None or self._paid is not None or self._epoch is not None:
            raise ValueError('Append cannot overlap another original grant')
        try:
            tail=self._read_tail()  # new LC work may exist; commit rechecks its actual bytes
            grant=_AppendGrant(_KEY,self,self._generation,tail);self._append=grant;return grant
        except BaseException as error:self._poison(str(error));raise

    def commit_append(self,grant):
        from . import nominal_trial_controller as c
        self._check()
        if type(grant) is not _AppendGrant or grant is not self._append or grant._session is not self or grant._used or grant._generation!=self._generation or grant._head!=_decode(self._state)['head'] or grant._batch_digest!=fingerprint(self._batch.binding):
            self._poison('invalid append grant');raise ValueError('Original unused append grant required')
        grant._used=True
        try:
            tail=self._read_tail();hour=self._generation;old=_decode(self._state)
            before=_decode(grant._before_tail)
            if tail[:len(before)]!=before:raise ValueError('Previously observed LC suffix bytes changed')
            appended=tail[len(before):]
            advances=[o for o in appended if o['kind']==c.ADVANCED]
            if len(appended)!=1 or len(advances)!=1:raise ValueError('Exactly one complete original source append required')
            op=advances[0];name=f'{hour:05d}.json';obj=self._observer
            entry=c._read(obj.directory/'events'/name)
            if (entry!=c._read(obj.mirror/'events'/name) or op!=c._read(obj.directory/'commits'/name)
                    or op!=c._read(obj.mirror/'commits'/name) or entry['previous']!=old['head']):
                raise ValueError('New source append dual anchors/previous head differ')
            publication=obj._publication(hour,self._raw_private)
            self._check_receipt_names(obj,hour+1)
            intent=c._read(obj.trial.paths['root']/'generator_publications'/f'{hour:05d}-intent.json')
            if (intent['physical_batch']!=self._batch.binding or intent['market_data'] is not False
                    or intent['logical_release']!=self._raw_private['schedule'][hour]['release_not_before']
                    or intent['physical_observation']['wall']!=intent['received_at']):
                raise ValueError('Original source publication allocation changed')
            c._check(intent['physical_observation'],old['physical_observation'])
            source={'hour':hour,'logical_at':self._raw_private['schedule'][hour]['release_not_before'],
                    'scheduled_gap':False,'publication_hash':fingerprint(publication)}
            if (entry['source']!=source or op['detail']['event_hash']!=fingerprint(entry)
                    or op['detail']['identity_hash']!=obj.digest):raise ValueError('Original append source/LC binding differs')
            physical=op['detail']['physical_observation'];wall=c._check(physical,old['physical_observation'])
            if c.stamp(publication['published_at'])>wall or c.stamp(publication['received_at'])>wall or op['at']!=physical['wall']:
                raise ValueError('Append physical publication order differs')
            batches=old['batch_ids']+[publication['receipt']['batch']]
            if {r[0] for r in self._runtime.feed.db.execute('SELECT id FROM batches')}!=set(batches):
                raise ValueError('Unregistered or incomplete feed append')
            names={f'{i:05d}.json' for i in range(hour+1)}
            for path in (obj.directory,obj.mirror):
                for folder in ('events','commits'):
                    if {p.name for p in (path/folder).glob('*.json')}!=names:raise ValueError('Extra/partial journal artifact')
            state={'events':hour+1,'head':fingerprint(op),'identity_hash':obj.digest,
                   'latest_binding':obj._prefix(hour,op,source),'logical_at':source['logical_at'],
                   'physical_observation':physical,'batch_ids':batches}
            self._check()
            # Commit private state only after EVERY actual suffix check succeeds.
            self._state=_bytes(state);self._sequence=op['seq'];self._tail=_bytes(op)
            self._prefixes=self._prefixes+(_bytes(state['latest_binding']),)
            self._generation=hour+1;self._append=None
            return _decode(_bytes(state['latest_binding']))
        except BaseException as error:self._poison(str(error));raise

    @contextmanager
    def append(self):
        grant=self.begin_append()
        try:
            yield grant
            self.commit_append(grant)
        except BaseException as error:self._poison(str(error));raise

    @contextmanager
    def read_grant(self,prefix,asof,*,months=3,context_hours=1234):
        from .quarantined_observation import snapshot_for_prefix
        self._check()
        if type(months) is not int or months!=3 or type(context_hours) is not int or context_hours!=1234:
            raise ValueError('Original three-month and complete1234-hour window required')
        if self._append is not None or self._read is not None:
            raise ValueError('Snapshot grant cannot overlap append/read')
        try:
            hour=prefix.get('hour') if type(prefix) is dict else None
            if type(hour) is not int or not 0<=hour<self._generation or _bytes(prefix)!=self._prefixes[hour]:
                raise ValueError('Read cannot borrow a prefix outside the pinned private generation')
            # Safe original fallback, preserves receipt/revision/as-of selection.
            # No raw-atom shortcut; no claim this scan is incremental yet.
            frame=snapshot_for_prefix(self._runtime.lifecycle,prefix,asof,months=months,context_hours=context_hours)
            if _bytes(frame.attrs.get('observation_prefix'))!=self._prefixes[hour]:
                raise ValueError('Original fallback returned another source generation')
            grant=_ReadGrant(_KEY,self,prefix,frame);self._read=grant
            yield grant
            grant._check()
        except BaseException as error:self._poison(str(error));raise
        finally:
            if self._read is not None:self._read._closed=True
            self._read=None

    @contextmanager
    def read_epoch(self):
        self._check()
        if self._epoch is not None or self._append is not None or self._read is not None or self._paid is not None:
            raise ValueError('Original read epoch cannot overlap another invocation')
        epoch=_ReadEpoch(_KEY,self);self._epoch=epoch
        try:
            yield epoch
            self._check()
            if self._epoch is not epoch or epoch.generation!=self._generation:
                raise ValueError('Pinned original read generation changed')
        except BaseException as error:self._poison(str(error));raise
        finally:epoch.close();self._epoch=None

    def has_read_epoch(self):
        self._check()
        return (not self._auditing and not self._building_snapshot
                and type(self._epoch) is _ReadEpoch and not self._epoch.closed)

    def snapshot_in_epoch(self,lifecycle,prefix,asof,*,months=3,context_hours=1234):
        from .quarantined_observation import snapshot_for_prefix
        from .v7_lifecycle import stamp
        self._check()
        if not self.has_read_epoch() or self._append is not None or self._read is not None:
            raise ValueError('Original pinned epoch required')
        if lifecycle.directory!=self._runtime.directory or lifecycle.plan!=self._runtime.lifecycle.plan:
            raise ValueError('Another lifecycle cannot borrow an actual receipt frame')
        if type(months) is not int or months!=3 or type(context_hours) is not int or context_hours!=1234:
            raise ValueError('Original three-month/1234-hour read scope required')
        epoch=self._epoch
        try:
            hour=prefix.get('hour') if type(prefix) is dict else None
            if type(hour) is not int or not 0<=hour<self._generation or _bytes(prefix)!=self._prefixes[hour]:
                raise ValueError('Unregistered original prefix cannot be read')
            key=(_bytes(prefix),stamp(asof).isoformat(),months,context_hours)
            if epoch.key!=key:
                if epoch.grant is not None:epoch.grant._closed=True
                epoch.grant=None;epoch.key=None
                self._building_snapshot=True
                try:
                    # Dispatcher sees no active session during this exact original
                    # disk fallback. No raw-atom construction or caller callback.
                    frame=snapshot_for_prefix(lifecycle,prefix,asof,months=months,context_hours=context_hours)
                finally:self._building_snapshot=False
                if (_bytes(frame.attrs.get('observation_prefix'))!=key[0]
                        or frame.attrs.get('snapshot_asof')!=key[1]
                        or frame.attrs.get('lifecycle_directory')!=str(self._runtime.directory)
                        or frame.attrs.get('feed_identity')!=self._runtime.feed.identity
                        or self._generation!=epoch.generation or self._epoch is not epoch):
                    raise ValueError('Original fallback changed the pinned actual read epoch')
                epoch.grant=_ReadGrant(_KEY,self,prefix,frame);epoch.key=key
            return epoch.grant.frame()
        except BaseException as error:self._poison(str(error));raise

    def validate_in_epoch(self,frame):
        attrs=frame.attrs
        if set(attrs)!={'availability_mode','feed_identity','snapshot_asof','lifecycle_directory',
                       'observation_prefix','months','context_hours'}:
            raise ValueError('Original actual frame provenance fields required')
        # Reconstruct if this one-frame-bounded capsule was replaced. Equality is
        # still to actual original receipt reconstruction at the fixed generation.
        self.snapshot_in_epoch(self._runtime.lifecycle,attrs['observation_prefix'],attrs['snapshot_asof'],
                               months=attrs['months'],context_hours=attrs['context_hours'])
        return self._epoch.grant.validate(frame)

    def _fresh_paid_audit(self,preparation):
        """Suspend ONLY the original same-runtime nominal read cache.

        The complete disk audit, private snapshot and original reserved lease
        are authority. A prior cached state is only an extra consistency check.
        The cooperative original controller lock is never released.
        """
        from . import nominal_trial_controller as controller
        self._check()
        if (controller is not self._controller or controller._READ is not self._cache_var
                or self._runtime.lifecycle.controller_lock() is not self._lock or not self._lock.is_locked):
            raise ValueError('Fresh audit requires the original nominal cache and held controller lock')
        original=preparation._live()
        scope=self._cache_var.get()
        if scope is not None:
            if (type(scope) is not dict or set(scope)!={'directory','state'}
                    or scope['directory']!=str(self._runtime.directory)
                    or scope['state'] is not None and type(scope['state']) is not dict):
                raise ValueError('Foreign or malformed nominal read scope cannot be suspended')
            if scope['state'] is not None and _bytes(scope['state'])!=self._state:
                raise ValueError('Original cached state differs from the private validated prefix')
        retained=None if scope is None else _bytes(scope)
        token=self._cache_var.set(None)
        try:
            fresh=self._full_audit(install=False)
            if (_bytes(fresh)!=self._state or self._cache_var.get() is not None
                    or scope is not None and _bytes(scope)!=retained):
                raise ValueError('Fresh disk audit/cache scope changed under its original lock')
            if original!=preparation._live():
                raise ValueError('Original paid reservation or lease changed during full audit')
            return fresh
        finally:
            # Even failure restores ContextVar nesting; the enclosing paid guard
            # durably poisons the session and consumes the original reservation.
            self._cache_var.reset(token)

    @contextmanager
    def paid_fit_guard(self,preparation):
        """Full source audit INSIDE the already reserved original <=300s lease."""
        from .forward_preparation import ForwardPreparation
        self._check()
        if (type(preparation) is not ForwardPreparation or preparation._runtime is not self._runtime
                or preparation.authority.directory!=self._trial.paths['authority']
                or self._append is not None or self._read is not None or self._paid is not None):
            raise ValueError('Original already reserved live preparation required')
        try:
            self._paid=preparation
            before=preparation._live()
            if before['status']!='reserved':raise ValueError('Preparation is not live reserved work')
            from .nominal_actual_trial import validate_preparation
            context=validate_preparation(self._runtime,preparation._header['job'],preparation._header['pipeline'])
            if context is None or context['role']!='original_nominal_actual_trial':
                raise ValueError('Original nominal paid job binding required')
            self._fresh_paid_audit(preparation);preparation._live()
            yield
            self._fresh_paid_audit(preparation);preparation._live()
        except BaseException as error:self._poison(str(error));preparation._fail(error);raise
        finally:self._paid=None


def active_original(runtime):
    """Future narrow hooks may find a live typed session, never a success bool.

    Full audits deliberately expose no cached grant. Current v2 hooks are not
    modified by this module; absence means the unchanged original path applies.
    """
    value=_ACTIVE.get()
    if value is None or value._auditing or value._building_snapshot:return None
    if type(value) is not NominalPrefixSession or value._runtime is not runtime:
        raise ValueError('Another runtime cannot borrow a nominal session')
    value._check()
    return value


def active_for_lifecycle(lifecycle):
    """Only original runtime/reader types in this active registered invocation."""
    value=_ACTIVE.get()
    if value is None or value._auditing or value._building_snapshot:return None
    from .v7_lifecycle import Lifecycle
    from .neural_forward_ticket import _LifecycleReader
    if type(value) is not NominalPrefixSession or type(lifecycle) not in (Lifecycle,_LifecycleReader):
        raise TypeError('Canonical original lifecycle reader required')
    if lifecycle.directory!=value._runtime.directory or lifecycle.plan!=value._runtime.lifecycle.plan:
        raise ValueError('Foreign lifecycle cannot enter the original read epoch')
    value._check();return value


def active_for_frame(frame):
    value=_ACTIVE.get()
    if value is None or value._auditing or value._building_snapshot:return None
    if type(value) is not NominalPrefixSession or frame.attrs.get('lifecycle_directory')!=str(value._runtime.directory):
        raise ValueError('Foreign frame cannot enter the original read epoch')
    value._check();return value


def preparation_guard(preparation):
    """Called only after original reserved _live; old routes stay unchanged."""
    from .forward_preparation import ForwardPreparation
    if type(preparation) is not ForwardPreparation:raise TypeError('Original preparation required')
    runtime=preparation._runtime
    session=active_original(runtime)
    if session is not None:return session.paid_fit_guard(preparation)
    from .nominal_trial_controller import from_lifecycle
    if from_lifecycle(runtime.lifecycle) is not None:
        raise ValueError('Registered nominal preparation requires its original prefix session')
    return nullcontext()
