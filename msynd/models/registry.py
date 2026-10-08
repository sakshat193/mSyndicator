"""Load active artifacts and atomically promote one model per role."""

from msynd.models.empirical import ModelArtifact as EmpiricalArtifact, provisional_models
from msynd.models.lgbm import ModelArtifact as QuantileArtifact


def active(conn):
    result = {role: (None, model) for role, model in provisional_models().items()}
    for id, role, kind, path in conn.execute("SELECT id,role,kind,path FROM ml.model WHERE active"):
        loader = {"empirical": EmpiricalArtifact, "lgbm_quantile": QuantileArtifact}.get(kind)
        if loader is None or role not in result:
            raise ValueError(f"unsupported active model {role}/{kind}")
        model = loader.load(path)
        if model.role != role:
            raise ValueError("artifact role differs from registry")
        result[role] = (id, model)
    return result


def promote(conn, model_id):
    with conn.transaction():
        # Serialize concurrent swaps before disabling the old model.
        conn.execute("LOCK TABLE ml.model IN SHARE ROW EXCLUSIVE MODE")
        row = conn.execute("SELECT role FROM ml.model WHERE id=%s", (model_id,)).fetchone()
        if row is None:
            raise ValueError(f"unknown model {model_id}")
        conn.execute("UPDATE ml.model SET active=false WHERE role=%s AND active", (row[0],))
        conn.execute("UPDATE ml.model SET active=true WHERE id=%s", (model_id,))
        conn.execute("SELECT pg_notify('model_changed',%s)", (str(model_id),))
