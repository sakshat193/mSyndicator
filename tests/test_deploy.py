import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture(scope='module')
def compose():
    docker = shutil.which('docker')
    if docker is None:
        pytest.skip('Docker CLI required to render Compose configuration')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [docker, 'compose', '--project-directory', str(root), '--env-file',
         str(root / '.env.example'), '--profile', 'app', '--profile', 'ops',
         'config', '--format', 'json'],
        env=os.environ | {'DUCKDNS_HOST': 'test.invalid', 'BASIC_AUTH_USER': 'test',
                         'BASIC_AUTH_HASH': '$2a$14$example'},
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_db_only_start_preserves_pilot_storage_and_port(compose):
    services = compose['services']
    assert {name for name, service in services.items() if not service.get('profiles')} == {'db'}
    assert services['db']['image'] == 'postgres:16'
    port, = services['db']['ports']
    assert (port['host_ip'], port['published'], port['target']) == ('127.0.0.1', '55432', 5432)
    volume, = services['db']['volumes']
    assert (volume['type'], volume['source'], volume['target']) == (
        'volume', 'pgdata', '/var/lib/postgresql/data')


def test_app_services_have_private_ports_and_bounded_logs(compose):
    for name in ('collector', 'engine', 'api', 'caddy'):
        service = compose['services'][name]
        assert service['profiles'] == ['app']
        assert service['restart'] == 'unless-stopped'
        assert service['healthcheck']['test']
        assert service['logging'] == {
            'driver': 'json-file', 'options': {'max-size': '10m', 'max-file': '5'}}
        if name != 'caddy':
            assert not service.get('ports')
            assert service['environment']['DATABASE_URL'] == 'postgresql://msynd:msynd@db:5432/msynd'
    caddy = compose['services']['caddy']
    assert {p['target'] for p in caddy['ports']} == {80, 443}
    # Compose re-escapes dollars when rendering reusable configuration.
    assert caddy['environment']['BASIC_AUTH_HASH'].replace('$$', '$') == '$2a$14$example'
    assert all(v.get('read_only') for v in caddy['volumes'] if v['target'] in ('/srv/web', '/srv/tiles', '/etc/caddy/Caddyfile'))


def test_commands_and_ops_do_not_start_the_pilot_database(compose):
    services = compose['services']
    assert services['collector']['command'] == ['python', '-m', 'msynd.collect']
    assert services['collector']['environment']['MMT_RATE'] == '0.2'
    assert services['collector']['environment']['MSYND_LIVE'] == '0'
    assert services['engine']['command'] == ['python', '-m', 'msynd.engine']
    assert services['api']['command'] == [
        'uvicorn', 'msynd.api.app:create_app', '--factory', '--host', '0.0.0.0', '--port', '8000']
    ops = services['ops']
    assert ops['profiles'] == ['ops']
    assert not ops.get('depends_on')
    assert ops['restart'] == 'no'
    assert ops['environment']['MSYND_LIVE'] == '0'
    assert {'/data', '/root/.oci'} <= {v['target'] for v in ops['volumes']}
