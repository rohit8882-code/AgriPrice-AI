.PHONY: help setup test lint format train evaluate run-api run-ui run-bot clean

PYTHON ?= python

help:
	@echo "Available commands:"
	@echo "  make setup      - Install base and dev dependencies"
	@echo "  make test       - Run full pytest test suite"
	@echo "  make lint       - Run ruff linter check"
	@echo "  make format     - Run black formatter"
	@echo "  make train      - Train model checkpoints"
	@echo "  make evaluate   - Run model evaluation benchmarks"
	@echo "  make run-api    - Run FastAPI backend service"
	@echo "  make run-ui     - Run Streamlit dashboard"
	@echo "  make run-bot    - Run Telegram advisory bot"
	@echo "  make clean      - Clean temporary cache and build artifacts"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements/dev.txt
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check src/ tests/

format:
	$(PYTHON) -m black src/ tests/

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

run-api:
	$(PYTHON) -m uvicorn agriprice.api.main:app --host 127.0.0.1 --port 8000 --reload

run-ui:
	$(PYTHON) -m streamlit run src/agriprice/dashboard/app.py

run-bot:
	$(PYTHON) -m agriprice.bot

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
