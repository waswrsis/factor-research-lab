"""Fixed public-only HTTP worker. No project/pandas/torch imports or subprocesses.

The parent's 30s whole-poll budget reserves 2s for cleanup; this worker gets the
remaining absolute wall deadline. Once initialized, its watchdog still fires
at that deadline if its parent dies (no immediate parent-death claim). A
successful result is emitted only after response/session cleanup.
"""
from __future__ import annotations

import datetime
import json
import os
import sys
import threading
import time

ENDPOINT = 'https://fapi.binance.com/fapi/v1/klines'


def transient_error(exc):
    """Narrow public transport classes only; never serialize bodies/messages."""
    import requests
    if isinstance(exc, (requests.exceptions.SSLError, requests.exceptions.ProxyError)):
        return None
    if isinstance(exc, (requests.Timeout, TimeoutError)):
        return {'kind': 'timeout'}
    if isinstance(exc, (requests.ConnectionError, requests.exceptions.ChunkedEncodingError)):
        return {'kind': 'connection'}
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if type(status) is int and (status == 429 or 500 <= status <= 599):
            return {'kind': 'http', 'status': status}
    return None


def fetch(params, deadline, *, session=None):
    import requests

    if (set(params) != {'symbol', 'interval', 'limit', 'endTime'}
            or not isinstance(params['symbol'], str) or not params['symbol'].isalnum()
            or not params['symbol'].isupper() or params['interval'] != '1h'
            or params['limit'] != 96 or type(params['endTime']) is not int):
        raise ValueError('Invalid bounded public request')
    remaining = deadline - time.time()
    if remaining > 30:
        raise ValueError('Public request clock moved backward')
    if remaining <= 0:
        raise TimeoutError('Public request deadline exceeded')
    client = session if session is not None else requests.Session()
    client.trust_env = False
    headers = {str(key).lower() for key in client.headers}
    if (headers & {'authorization', 'proxy-authorization', 'cookie', 'x-mbx-apikey', 'x-api-key'}
            or any(getattr(client, key, None) for key in ('auth', 'cookies', 'params', 'proxies'))):
        if session is None:
            client.close()
        raise ValueError('Public request cannot carry credentials or overrides')
    try:
        response = client.get(ENDPOINT, params=params, stream=True, allow_redirects=False,
                              timeout=(min(5, remaining), min(15, remaining)))
        try:
            response.raise_for_status()
            if 300 <= response.status_code < 400:
                raise ValueError('Public endpoint cannot redirect')
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                if time.time() >= deadline:
                    raise TimeoutError('Public request deadline exceeded')
                size += len(chunk)
                if size > 1_048_576:
                    raise ValueError('Public response exceeds one MiB')
                chunks.append(chunk)
            payload = json.loads(b''.join(chunks))
            received = datetime.datetime.now(datetime.UTC).isoformat()
        finally:
            response.close()
    finally:
        if session is None:
            client.close()
    if time.time() >= deadline:
        raise TimeoutError('Public request cleanup exhausted deadline')
    return {'payload': payload, 'received_at': received}


def main():
    deadline = float(sys.argv[1])
    remaining = deadline - time.time()
    if not 0 < remaining <= 30:
        raise TimeoutError('Invalid public worker deadline')
    watchdog = threading.Timer(remaining, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    # Leave watchdog active until process exit, including output pipe writes.
    raw = sys.stdin.buffer.read(4097)
    if len(raw) > 4096:
        raise ValueError('Public request manifest too large')
    try:
        result = fetch(json.loads(raw), deadline)
    except Exception as exc:
        error = transient_error(exc)
        if error is None:
            raise
        result = {'error': error}
    sys.stdout.write(json.dumps(result, allow_nan=False, separators=(',', ':')))
    sys.stdout.flush()


if __name__ == '__main__':
    main()
