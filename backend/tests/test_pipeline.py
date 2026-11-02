"""Testes minimos do pipeline: seed, ETL, holdout e os dois modelos."""
import pandas as pd

from app import models
from app.etl.pipeline import run_etl
from app.ml.anomaly import detect_anomalies
from app.ml.failure_prediction import FEATURES, build_dataset, evaluate_holdout, predict_failures
from seed import EVENTS_PER_ASSET, ASSETS, synthetic_events


def _seed_df() -> pd.DataFrame:
    records = [
        {**ev, "asset_id": i}
        for i, (_, events) in enumerate(synthetic_events(), start=1)
        for ev in events
    ]
    return build_dataset(records)


def test_seed_e_deterministico():
    a = [ev["temperature"] for _, evs in synthetic_events() for ev in evs]
    b = [ev["temperature"] for _, evs in synthetic_events() for ev in evs]
    assert a == b
    assert len(a) == len(ASSETS) * EVENTS_PER_ASSET


def test_so_ativos_doentes_tem_falha():
    for spec, events in synthetic_events():
        downs = sum(ev["status"] == "down" for ev in events)
        if spec["healthy"]:
            assert downs == 0
        else:
            assert downs > 0


def test_holdout_estratificado_e_metricas_no_intervalo():
    df = _seed_df()
    ev = evaluate_holdout(df)
    assert ev["evaluated"] is True
    assert ev["train_size"] + ev["test_size"] == len(df)
    # Estratificado: a taxa de falha no teste fica perto da taxa global.
    rate_all = df["failed"].mean()
    rate_test = ev["test_positives"] / ev["test_size"]
    assert abs(rate_test - rate_all) < 0.03
    for k in ("roc_auc", "precision", "recall"):
        assert 0.0 <= ev[k] <= 1.0
    cm = ev["confusion_matrix"]
    assert sum(cm.values()) == ev["test_size"]
    assert cm["tp"] + cm["fn"] == ev["test_positives"]


def test_holdout_e_reprodutivel():
    df = _seed_df()
    assert evaluate_holdout(df) == evaluate_holdout(df)


def test_holdout_recusa_classe_rara_pequena():
    df = _seed_df()
    poucos = pd.concat([df[df["failed"] == 0], df[df["failed"] == 1].head(3)])
    ev = evaluate_holdout(poucos)
    assert ev["evaluated"] is False


def test_etl_gera_metricas_e_e_idempotente(seeded_db):
    r1 = run_etl(seeded_db)
    n1 = seeded_db.query(models.Metric).count()
    r2 = run_etl(seeded_db)
    n2 = seeded_db.query(models.Metric).count()
    assert r1["processed_events"] == len(ASSETS) * EVENTS_PER_ASSET
    assert n1 == n2 > 0
    assert r1 == r2


def test_predict_failures_pontua_todos_os_ativos_e_devolve_avaliacao(seeded_db):
    out = predict_failures(seeded_db)
    assert out["trained"] is True
    assert out["assets_scored"] == len(ASSETS)
    assert set(out["feature_importances"]) == set(FEATURES)
    assert out["evaluation"]["evaluated"] is True
    preds = seeded_db.query(models.Prediction).filter_by(target="failure_risk").all()
    assert len(preds) == len(ASSETS)
    # Rodar de novo substitui, nao duplica.
    predict_failures(seeded_db)
    assert seeded_db.query(models.Prediction).filter_by(target="failure_risk").count() == len(ASSETS)


def test_predict_failures_recusa_pouco_dado(db):
    assert predict_failures(db)["trained"] is False


def test_detect_anomalies_roda_no_seed(seeded_db):
    out = detect_anomalies(seeded_db)
    assert out["trained"] is True
