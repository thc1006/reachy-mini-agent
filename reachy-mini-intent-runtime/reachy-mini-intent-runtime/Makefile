.PHONY: install verify test lint format demo

install:
	pip install -e '.[dev]'

verify:
	./scripts/verify.sh

test:
	python -m pytest -q

lint:
	ruff check .

format:
	ruff format .

demo:
	python -m reachy_intent_runtime.demo --script demo/hospital_interrupt_scenario.json
