import math
import os
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
import pytest

from msynd.db import migrate

pytestmark = [pytest.mark.slow, pytest.mark.skipif(
    not os.environ.get('MSYND_REPLAY_DAY'), reason='set MSYND_REPLAY_DAY to a recorded day directory')]


def test_recorded_replay_day():
    from msynd.ops.replay_day import replay

    admin_url = os.environ['MSYND_TEST_DSN']
    name = f'msynd_test_replay_day_{os.getpid()}'
    target = urlunsplit(urlsplit(admin_url)._replace(path=f'/{name}'))
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL('CREATE DATABASE {}').format(sql.Identifier(name)))
        try:
            migrate.apply(target)
            result = replay(os.environ['MSYND_REPLAY_DAY'], target, speed=60)
            assert result['invalid_share'] < .01
            assert result['replay_errors'] == 0
            assert result['runs_unframed'] == 0
            assert result['a1']['minutes'] > 0
            assert math.isfinite(result['a6']['p95'])
        finally:
            admin.execute(sql.SQL('DROP DATABASE {} WITH (FORCE)').format(sql.Identifier(name)))
