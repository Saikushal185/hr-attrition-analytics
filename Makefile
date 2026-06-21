.PHONY: install test lint run dash

install:
	pip install -r requirements-dev.txt

test:
	pytest

lint:
	ruff check src tests

run:
	python -m src.main

dash:
	streamlit run dashboard/app.py
