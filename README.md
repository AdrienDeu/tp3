# TP3 — Régler, juger… et ne pas se mentir (NLU, porte PASS/FAIL)

```
make setup                 # MLflow + RustFS (comme au TP2)
uv sync
uv run python main.py      # RÉGLER : entraîne + mesure sur la VALIDATION (ne lit pas le test)
make prompt-exemples       # le prompt système, réglé lui aussi sur la validation
make juger                 # JUGER : porte de qualité sur le TEST SCELLÉ (une fois)
```

- MLflow : http://localhost:5000 (bascule sur **Model training** en haut à gauche)
- RustFS : http://localhost:9001 (`rustfsadmin` / `rustfsadmin`)

Guide pas à pas : **TP_GUIDE_TP3.md**.
