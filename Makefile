ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /c
PY := creditS/Scripts/python.exe
ifneq ($(wildcard .runtime/python311/tools/python.exe),)
BOOTSTRAP := ".runtime\python311\tools\python.exe"
else
BOOTSTRAP := py -3.11
endif
else
PY := creditS/bin/python
BOOTSTRAP := python3.11
endif

.PHONY: setup data train test lint mlflow fairness promote up down demo
setup:
ifeq ($(OS),Windows_NT)
	@if not exist "$(PY)" $(BOOTSTRAP) -m venv creditS
else
	@test -x "$(PY)" || $(BOOTSTRAP) -m venv creditS
endif
	"$(PY)" scripts/setup.py

data:
	"$(PY)" scripts/dev.py data
train:
	"$(PY)" scripts/mlflow_server.py
	"$(PY)" scripts/dev.py train
mlflow:
	"$(PY)" scripts/mlflow_server.py
fairness:
	"$(PY)" scripts/mlflow_server.py
	"$(PY)" -m dvc repro explain
promote:
	"$(PY)" scripts/mlflow_server.py
	"$(PY)" -m src.models.promote
test:
	"$(PY)" scripts/dev.py test
lint:
	"$(PY)" scripts/dev.py lint
up:
	"$(PY)" scripts/dev.py up
down:
	"$(PY)" scripts/dev.py down
demo:
	"$(PY)" scripts/dev.py demo


