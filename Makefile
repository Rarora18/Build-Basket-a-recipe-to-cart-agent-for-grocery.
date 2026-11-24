PY=.venv/bin/python

.PHONY: seed train test eval serve

seed:
	$(PY) scripts/generate_seed.py

train:
	$(PY) scripts/train.py

test:
	$(PY) -m pytest -q

eval:
	$(PY) scripts/eval_all.py

serve:
	$(PY) -m uvicorn basket.api:app --host 0.0.0.0 --port 8000
