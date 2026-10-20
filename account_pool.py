"""Select CLI-managed subscriptions; persist no credentials or quota state."""
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

_CACHE = {}
_LOCK = threading.Lock()


class SubscriptionQuotaRejected(RuntimeError):
    """A rejected request, before any output, confirmed exhausted by usage API."""


def configured_accounts():
    from hermes_cli.config import load_config
    rows = load_config().get('claude_subscription_accounts') or []
    if not isinstance(rows, list):
        raise ValueError('claude_subscription_accounts must be a list')
    result, paths, labels = [], set(), set()
    for row in rows:
        label, directory = row.get('label'), row.get('config_dir')
        if not isinstance(label, str) or not label or not isinstance(directory, str):
            raise ValueError('Each Claude subscription needs label and config_dir')
        directory = str(Path(directory).expanduser().resolve())
        if directory in paths or label in labels:
            raise ValueError('Duplicate Claude subscription label or config directory')
        paths.add(directory)
        labels.add(label)
        result.append({'label': label, 'config_dir': directory})
    return result


def usage(account, *, fresh=False):
    """Read only the OAuth usage endpoint. The official CLI alone refreshes tokens."""
    path = Path(account['config_dir']) / '.credentials.json'
    try:
        key = (str(path), path.stat().st_mtime_ns)
        token = json.loads(path.read_text())['claudeAiOauth']['accessToken']
        with _LOCK:
            cached = _CACHE.get(key)
            if not fresh and cached and time.monotonic() - cached[0] < 30:
                return cached[1]
        request = urllib.request.Request('https://api.anthropic.com/api/oauth/usage', headers={
            'Authorization': 'Bearer ' + token, 'anthropic-beta': 'oauth-2025-04-20',
            'User-Agent': 'claude-code/2.1.0', 'Accept': 'application/json'})
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.load(response)
        if not isinstance(payload, dict):
            raise ValueError('Invalid usage response')
        with _LOCK:
            _CACHE[key] = (time.monotonic(), payload)
        return payload
    except Exception as exc:
        # Never include HTTP bodies, headers, credential contents, or exception reprs.
        raise RuntimeError('Claude usage unavailable: ' + type(exc).__name__) from None


def exhausted(payload, model):
    keys = ['five_hour', 'seven_day']
    if 'opus' in model:
        keys.append('seven_day_opus')
    if 'sonnet' in model:
        keys.append('seven_day_sonnet')
    return any(isinstance((payload.get(k) or {}).get('utilization'), (float, int))
               and payload[k]['utilization'] >= 100 for k in keys)


def eligible(payload, model):
    # No paid overflow: a successful usage read must explicitly report it disabled.
    if (payload.get('extra_usage') or {}).get('is_enabled') is not False:
        return False
    for key in ('five_hour', 'seven_day'):
        value = (payload.get(key) or {}).get('utilization')
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 100:
            return False
    return not exhausted(payload, model)


def candidates(accounts, model, preferred=None):
    ordered = sorted(accounts, key=lambda a: a['label'] != preferred) if preferred else accounts
    available, reasons = [], []
    for account in ordered:
        try:
            payload = usage(account)
            if eligible(payload, model):
                available.append(account)
            else:
                reasons.append(account['label'] + ': exhausted or extra usage not disabled')
        except RuntimeError:
            reasons.append(account['label'] + ': usage unavailable')
    if not available:
        raise RuntimeError('No included Claude subscription available; ' + '; '.join(reasons))
    return available


def usage_snapshot():
    from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
    from datetime import datetime, timezone
    windows, details = [], []
    for account in configured_accounts():
        try:
            payload = usage(account, fresh=True)
            for key, name in [('five_hour', '5-hour'), ('seven_day', 'weekly'),
                              ('seven_day_opus', 'Opus weekly'), ('seven_day_sonnet', 'Sonnet weekly')]:
                row = payload.get(key) or {}
                if isinstance(row.get('utilization'), (int, float)):
                    reset = row.get('resets_at')
                    windows.append(AccountUsageWindow(label=account['label'] + ' ' + name,
                        used_percent=row['utilization'],
                        reset_at=datetime.fromisoformat(reset.replace('Z', '+00:00')) if reset else None))
            details.append(account['label'] + ': extra usage ' +
                           ('OFF' if (payload.get('extra_usage') or {}).get('is_enabled') is False else 'ON/UNKNOWN — not admitted'))
        except RuntimeError:
            details.append(account['label'] + ': usage unavailable')
    return AccountUsageSnapshot(provider='claude-subscription-directsdk-experimental',
        source='oauth_usage_api', fetched_at=datetime.now(timezone.utc), title='Claude subscription pool',
        windows=windows, details=details)
