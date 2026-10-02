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

.PHONY: setup data train test lint up down demo
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
	"$(PY)" scripts/dev.py train
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


