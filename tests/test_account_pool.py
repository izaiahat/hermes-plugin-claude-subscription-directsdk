"""Local fixtures validate contracts, not real provider quota rejection."""
import json
from pathlib import Path
import pytest
import account_pool as pool
import directsdk

ACCOUNTS = [{'label': 'one', 'config_dir': '/one'}, {'label': 'two', 'config_dir': '/two'}]

@pytest.fixture(autouse=True)
def pool_accounts(tmp_path, monkeypatch):
    monkeypatch.setattr(pool, 'configured_accounts', lambda: [])
    for account in ACCOUNTS:
        directory = tmp_path/account['label']
        directory.mkdir()
        monkeypatch.setitem(account, 'config_dir', str(directory))


def payload(used=0, extra=False):
    return {'five_hour': {'utilization': used}, 'seven_day': {'utilization': used}, 'extra_usage': {'is_enabled': extra}}

def test_selection_and_reset(monkeypatch):
    state = {'one': payload(100), 'two': payload()}
    calls = []
    def once(request, kwargs, *args):
        calls.append(args[-1]['label'])
        yield 'complete'
    client, request = run_wrapper(monkeypatch, once)
    monkeypatch.setattr(pool, 'usage', lambda a, **kw: state[a['label']])
    list(client._run(request, {'model': 'opus'}, '', [], set(), '', []))
    assert calls == ['two']
    state['one'] = payload()
    client._pool_label = None
    list(client._run(directsdk.Request(client), {'model': 'opus'}, '', [], set(), '', []))
    assert calls == ['two', 'one']
    client._pool_label = 'two'
    list(client._run(directsdk.Request(client), {'model': 'opus'}, '', [], set(), '', []))
    assert calls[-1] == 'two'

def test_no_paid_or_unknown_fallback(monkeypatch):
    def once(*args):
        pytest.fail('no eligible account may launch native')
        yield
    client, request = run_wrapper(monkeypatch, once)
    monkeypatch.setattr(pool, 'usage', lambda a, **kw: payload(extra=True))
    with pytest.raises(RuntimeError, match='no paid fallback'):
        list(client._run(request, {'model': 'opus'}, '', [], set(), '', []))
    assert not pool.eligible({}, 'opus')

def run_wrapper(monkeypatch, once):
    monkeypatch.delenv('CLAUDE_CONFIG_DIR', raising=False)
    monkeypatch.delenv('CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIR', raising=False)
    monkeypatch.setattr(pool, 'configured_accounts', lambda: ACCOUNTS)
    monkeypatch.setattr(pool, 'usage', lambda a, **kw: payload())
    client = directsdk.Client()
    monkeypatch.setattr(client, '_run_once', once)
    request = directsdk.Request(client)
    client._requests.add(request)
    return client, request

def test_rejected_handoff_preserves_history(monkeypatch):
    calls = []
    def once(request, kwargs, *args):
        account = args[-1]
        calls.append((account['label'], kwargs['messages']))
        if account['label'] == 'one':
            raise pool.SubscriptionQuotaRejected('fixture quota')
        yield 'complete'
    client, request = run_wrapper(monkeypatch, once)
    history = [{'role': 'user', 'content': 'same task'}]
    assert list(client._run(request, {'model': 'opus', 'messages': history}, '', [], set(), '', [])) == ['complete']
    assert calls == [('one', history), ('two', history)]
    assert not client._requests
    assert client._pool_label == 'two'

def test_no_replay_after_output(monkeypatch):
    calls = []
    def once(request, kwargs, *args):
        calls.append(args[-1]['label'])
        yield 'partial'
        raise pool.SubscriptionQuotaRejected('fixture quota')
    client, request = run_wrapper(monkeypatch, once)
    with pytest.raises(pool.SubscriptionQuotaRejected):
        list(client._run(request, {'model': 'opus'}, '', [], set(), '', []))
    assert calls == ['one']

def test_cancel_does_not_rotate(monkeypatch):
    calls = []
    def once(request, kwargs, *args):
        calls.append(args[-1]['label'])
        request.cancel()
        raise pool.SubscriptionQuotaRejected('fixture quota')
        yield
    client, request = run_wrapper(monkeypatch, once)
    with pytest.raises(pool.SubscriptionQuotaRejected):
        list(client._run(request, {'model': 'opus'}, '', [], set(), '', []))
    assert calls == ['one']

def test_reports_both_accounts(monkeypatch):
    monkeypatch.setattr(pool, 'configured_accounts', lambda: ACCOUNTS)
    monkeypatch.setattr(pool, 'usage', lambda a, **kw: payload(25))
    snapshot = pool.usage_snapshot()
    assert [w.label for w in snapshot.windows] == ['one 5-hour', 'one weekly', 'two 5-hour', 'two weekly']
    assert all(w.used_percent == 25 for w in snapshot.windows)
    assert snapshot.details == ['one: extra usage OFF', 'two: extra usage OFF']

def test_fresh_admission_on_handoff(monkeypatch):
    calls = []
    state = {'one': payload(), 'two': payload()}
    def once(request, kwargs, *args):
        calls.append(args[-1]['label'])
        state['two'] = payload(extra=True)
        raise pool.SubscriptionQuotaRejected('fixture quota')
        yield
    client, request = run_wrapper(monkeypatch, once)
    monkeypatch.setattr(pool, 'usage', lambda a, **kw: state[a['label']])
    with pytest.raises(RuntimeError, match='no paid fallback'):
        list(client._run(request, {'model': 'opus'}, '', [], set(), '', []))
    assert calls == ['one']


def test_overrides_cannot_bypass_pool(monkeypatch):
    def once(*args):
        pytest.fail('override must not launch native')
        yield
    client, request = run_wrapper(monkeypatch, once)
    for overrides in ({'CLAUDE_CONFIG_DIR': '/pin'}, {'OTHER': 'value'}):
        client.env = overrides
        with pytest.raises(ValueError, match='remove custom env'):
            list(client._run(request, {'model': 'opus'}, '', [], set(), '', []))


def test_official_renewal_then_preflight(monkeypatch):
    import urllib.error
    a = ACCOUNTS[1]
    path = Path(a['config_dir'])/'.credentials.json'
    path.write_text(json.dumps({'claudeAiOauth': {'accessToken': 'fixture-expired'}}))
    seen = []
    def fetch(token):
        seen.append(token)
        if token == 'fixture-expired':
            raise urllib.error.HTTPError('https://fixture.invalid', 401, 'fixture', {}, None)
        return payload()
    def renew(account):
        assert account == a
        path.write_text(json.dumps({'claudeAiOauth': {'accessToken': 'fixture-renewed'}}))
    monkeypatch.setattr(pool, '_usage_request', fetch)
    monkeypatch.setattr(pool, '_renew_login', renew)
    assert pool.eligible(pool.usage(a, fresh=True), 'opus')
    assert seen == ['fixture-expired', 'fixture-renewed']


def test_account_lock_serializes_and_cancels(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered = threading.Event()
    cancelled = threading.Event()
    def wait():
        entered.set()
        with pool.account_lock(ACCOUNTS[0], cancelled=cancelled):
            pytest.fail('must not pass the held lock')
    with pool.account_lock(ACCOUNTS[0]):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(wait)
            assert entered.wait(2)
            cancelled.set()
            with pytest.raises(RuntimeError, match='cancelled'):
                future.result(timeout=2)


def test_http_quota_rejection_retries_once(monkeypatch, tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import sys
    import threading
    from test_directsdk_admission import NATIVE
    from admission import Admission
    calls = []
    state = {'one': payload(), 'two': payload()}
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            if len(calls) == 1:
                state['one'] = payload(100)
                body = b'{"error":{"message":"fixture subscription exhausted"}}'
                self.send_response(429)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            events = [
                {'type':'message_start','message':{'id':'ok','role':'assistant','model':'sonnet','content':[], 'usage':{'input_tokens':1,'output_tokens':1}}},
                {'type':'content_block_start','index':0,'content_block':{'type':'text','text':'SUCCESS'}},
                {'type':'content_block_stop','index':0},
                {'type':'message_delta','delta':{'stop_reason':'end_turn'},'usage':{}},
                {'type':'message_stop'}]
            self.wfile.write(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    for key in ('CLAUDE_CONFIG_DIR', 'CLAUDE_SUBSCRIPTION_DIRECTSDK_CONFIG_DIR', 'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(pool, 'configured_accounts', lambda: ACCOUNTS)
    monkeypatch.setattr(pool, 'usage', lambda a, **kw: state[a['label']])
    monkeypatch.setattr(directsdk, 'Admission', lambda upstream, timeout: Admission(f'http://127.0.0.1:{peer.server_port}', timeout))
    native = tmp_path/'native.py'
    native.write_text(NATIVE)
    client = directsdk.Client(command=[sys.executable, str(native)])
    try:
        result = client.create(model='sonnet', messages=[{'role':'user','content':'fixture'}])
        assert result.choices[0].message.content == 'SUCCESS'
        assert result.usage.native_account == 'two'
        assert len(calls) == 2
        assert not client._requests
    finally:
        client.close()
        peer.shutdown()
        thread.join()
        peer.server_close()
