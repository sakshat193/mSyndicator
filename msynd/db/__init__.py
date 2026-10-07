import psycopg

from msynd.config import settings


def connect(url=None):
    return psycopg.connect(url or settings.database_url, connect_timeout=5)
