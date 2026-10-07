import os
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
import pytest


def pytest_collection_modifyitems(items):
    if os.environ.get("MSYND_LIVE") != "1":
        for item in items:
            if "live" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="live traffic requires explicit approval"))


@pytest.fixture(scope="session")
def db_url():
    from msynd.db import migrate

    admin_url = os.environ.get("MSYND_TEST_DSN", "postgresql://msynd:msynd@localhost:55432/postgres")
    name = f"msynd_test_{os.getpid()}"
    parts = urlsplit(admin_url)
    url = urlunsplit(parts._replace(path=f"/{name}"))
    with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        try:
            migrate.apply(url)
            yield url
        finally:
            admin.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


@pytest.fixture
def db(db_url):
    with psycopg.connect(db_url, connect_timeout=5) as conn:
        try:
            yield conn
        finally:
            conn.rollback()
