install:
	python3 -m pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check src tests

run:
	python3 -m market_regime.main --ticker AAPL

research:
	python3 -m market_regime.research

clean:
	rm -rf dist build *.egg-info
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
