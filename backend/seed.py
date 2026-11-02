"""Popula o banco com dados sintéticos pra demonstração.

Gera ativos saudáveis e alguns "doentes" (com piora ao longo do tempo e
eventos de falha), assim os modelos de ML têm sinal pra aprender.

    python seed.py

Os dados são 100% sintéticos. O status de cada leitura sai de uma regra fixa
sobre temperatura e vibração (ver `synthetic_events`), e isso pesa na leitura
de qualquer métrica de modelo feita sobre eles: ver `avaliar.py`.
"""
import random
from datetime import datetime, timedelta

from app.database import Base, SessionLocal, engine
from app import models

# Semente fixa: a demonstração precisa dar o mesmo banco toda vez, senão os
# números do README e os do dashboard nunca batem.
SEED = 42

ASSETS = [
    {"name": "Compressor A", "type": "compressor", "location": "Planta 1", "healthy": True},
    {"name": "Bomba Hidráulica B", "type": "pump", "location": "Planta 1", "healthy": True},
    {"name": "Motor C", "type": "motor", "location": "Planta 2", "healthy": False},
    {"name": "Turbina D", "type": "turbine", "location": "Planta 2", "healthy": False},
    {"name": "Esteira E", "type": "conveyor", "location": "Planta 3", "healthy": True},
]

EVENTS_PER_ASSET = 60


def synthetic_events(seed: int = SEED, now: datetime | None = None):
    """Gera as leituras sintéticas, ativo por ativo, sem tocar no banco.

    Devolve uma lista de (spec_do_ativo, [leituras]). Cada leitura é um dict
    com as colunas de `models.Event` menos `asset_id`. Separado do banco para
    que `avaliar.py` e os testes usem exatamente os mesmos dados do seed.
    """
    rng = random.Random(seed)
    now = now or datetime.utcnow()
    out = []
    for spec in ASSETS:
        events = []
        runtime = rng.uniform(1000, 5000)
        for i in range(EVENTS_PER_ASSET):
            runtime += rng.uniform(0.5, 2.0)

            if spec["healthy"]:
                temp = rng.gauss(70, 5)
                vib = rng.gauss(4.0, 0.8)
                press = rng.gauss(9.0, 0.6)
                status = "ok"
            else:
                drift = i / EVENTS_PER_ASSET  # vai piorando com o tempo
                temp = rng.gauss(78 + drift * 18, 4)
                vib = rng.gauss(5.0 + drift * 4, 1.0)
                press = rng.gauss(10.0 + drift * 2, 0.8)
                # O rótulo é uma regra determinística sobre duas das features.
                if temp > 92 or vib > 9:
                    status = "down"
                elif temp > 85 or vib > 7:
                    status = "warning"
                else:
                    status = "ok"

            events.append({
                "temperature": round(temp, 2),
                "vibration": round(vib, 2),
                "pressure": round(press, 2),
                "runtime_hours": round(runtime, 1),
                "status": status,
                "timestamp": now - timedelta(hours=(EVENTS_PER_ASSET - i)),
            })
        out.append((spec, events))
    return out


def populate(db, seed: int = SEED) -> None:
    """Grava os ativos e as leituras sintéticas na sessão recebida."""
    for spec, events in synthetic_events(seed):
        asset = models.Asset(name=spec["name"], type=spec["type"], location=spec["location"])
        db.add(asset)
        db.flush()  # precisa do asset.id antes de criar os eventos
        for ev in events:
            db.add(models.Event(asset_id=asset.id, **ev))
    db.commit()


def generate():
    # DERRUBA e recria tudo. O seed é para demonstração, não para completar
    # base existente: apontado para o Postgres do compose, apaga os dados.
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    populate(db)
    print(f"Seed ok: {db.query(models.Asset).count()} ativos, {db.query(models.Event).count()} eventos")
    db.close()


if __name__ == "__main__":
    generate()
