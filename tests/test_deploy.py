import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


def render_compose(*files, **overrides):
    docker = shutil.which('docker')
    if docker is None:
        pytest.skip('Docker CLI required to render Compose configuration')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [docker, 'compose', '--project-directory', str(root), '--env-file',
         str(root / '.env.example'),
         *[arg for file in files or ('compose.yaml',) for arg in ('-f', str(root / file))],
         '--profile', 'app', '--profile', 'ops',
         'config', '--format', 'json'],
        env={key: value for key, value in os.environ.items()
             if key not in ('COMPOSE_FILE', 'MMT_POLL_BUDGET', 'MMT_HARVEST_BUDGET')}
        | {'POSTGRES_PASSWORD': 'testpw', 'DUCKDNS_HOST': 'test.invalid', 'BASIC_AUTH_USER': 'test',
           'BASIC_AUTH_HASH': '$2a$14$example'} | overrides,
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


@pytest.fixture(scope='module')
def compose():
    return render_compose()


def test_db_only_start_preserves_pilot_storage_and_port(compose):
    services = compose['services']
    assert {name for name, service in services.items() if not service.get('profiles')} == {'db'}
    assert services['db']['image'] == 'postgres:16'
    assert services['db']['environment']['POSTGRES_PASSWORD'] == 'testpw'
    port, = services['db']['ports']
    assert (port['host_ip'], port['published'], port['target']) == ('127.0.0.1', '55432', 5432)
    volume, = services['db']['volumes']
    assert (volume['type'], volume['source'], volume['target']) == (
        'volume', 'pgdata', '/var/lib/postgresql/data')


def test_database_recovers_after_host_restart(compose):
    assert compose['services']['db']['restart'] == 'unless-stopped'


def test_direct_collector_build_is_isolated_from_backup_runtime(compose):
    collector = compose['services']['collector']
    ops = compose['services']['ops']
    assert collector['image'] != ops['image']
    assert collector['build']['args']['PYTHON_BASE'] == 'python:3.12-slim-bookworm'
    assert 'PYTHON_BASE' not in ops['build'].get('args', {})
    assert 'MSYND_RELAY' not in collector['environment']
    assert 'network_mode' not in collector


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
            assert service['environment']['DATABASE_URL'] == 'postgresql://msynd:testpw@db:5432/msynd'
    caddy = compose['services']['caddy']
    assert {p['target'] for p in caddy['ports']} == {80, 443}
    # Compose re-escapes dollars when rendering reusable configuration.
    assert caddy['environment']['BASIC_AUTH_HASH'].replace('$$', '$') == '$2a$14$example'
    assert all(v.get('read_only') for v in caddy['volumes'] if v['target'] in ('/srv/web', '/srv/tiles', '/etc/caddy/Caddyfile'))


def test_commands_and_ops_do_not_start_the_pilot_database(compose):
    services = compose['services']
    assert services['collector']['command'] == [
        'python', '-m', 'msynd.collect', '--budget', 'poll=5000', '--budget', 'harvest=0']
    assert services['collector']['environment']['MMT_RATE'] == '0.2'
    assert services['collector']['environment']['MSYND_LIVE'] == '0'
    assert services['engine']['command'] == ['python', '-m', 'msynd.engine']
    assert services['api']['command'] == [
        'uvicorn', 'msynd.api.app:create_app', '--factory', '--host', '0.0.0.0', '--port', '8000']
    ops = services['ops']
    assert ops['profiles'] == ['ops']
    assert not ops.get('depends_on')
    assert ops['restart'] == 'no'
    targets = {volume['target'] for volume in ops['volumes']}
    assert {'/data', '/root/.config/rclone'} <= targets
    assert '/root/.oci' not in targets
    assert 'BACKUP_REMOTE' in ops['environment']
    assert ops['environment']['MSYND_LIVE'] == '0'


def test_ops_backup_configuration():
    root = Path(__file__).resolve().parents[1]
    ops = (root / 'compose.yaml').read_text().split('\n  ops:\n', 1)[1].split('\nvolumes:', 1)[0]
    assert '      BACKUP_REMOTE: ${BACKUP_REMOTE:-}\n' in ops
    assert '      - /data:/data\n' in ops
    assert '      - ${RCLONE_CONFIG_DIR:-/data/rclone}:/root/.config/rclone' in ops.splitlines()
    assert '/root/.oci' not in ops and 'OCI_' not in ops
    assert 'BACKUP_REMOTE=\nRCLONE_CONFIG_DIR=/data/rclone\n' in (root / '.env.example').read_text()


def test_dockerfile_installs_rclone():
    dockerfile = (Path(__file__).resolve().parents[1] / 'Dockerfile').read_text()
    install, = [line for line in dockerfile.splitlines() if 'apt-get install ' in line]
    assert {'postgresql-client', 'rclone'} <= set(install.split())


@pytest.mark.parametrize('budgets, expected', [
    ({}, ['poll=5000', 'harvest=0']),
    ({'MMT_POLL_BUDGET': '1200', 'MMT_HARVEST_BUDGET': '500'}, ['poll=1200', 'harvest=500']),
])
def test_collector_budget_interpolation(budgets, expected):
    collector = render_compose(**budgets)['services']['collector']
    assert collector['command'] == [
        'python', '-m', 'msynd.collect', '--budget', expected[0], '--budget', expected[1]]


def test_relay_override_uses_host_network_and_loopback_database():
    config = render_compose('compose.yaml', 'compose.relay.yaml', POSTGRES_PASSWORD='relaypw')
    collector = config['services']['collector']
    assert collector['network_mode'] == 'host'
    assert not collector.get('networks')
    assert collector['environment']['DATABASE_URL'] == 'postgresql://msynd:relaypw@127.0.0.1:55432/msynd'
    assert collector['environment']['MSYND_RELAY'] == 'http://127.0.0.1:18080'
    assert 'MSYND_PROXY' not in collector['environment']
    assert config['services']['engine']['environment']['DATABASE_URL'] == 'postgresql://msynd:relaypw@db:5432/msynd'
