.PHONY: install test generate validate power scaffold clean

install:
	pip install -r requirements.txt

test:
	python3 tests/test_pipeline.py

generate:
	python3 -m psba.cli generate --languages en --paraphrases 3 --orders 2 --out data/items.jsonl

validate:
	python3 -m psba.cli validate --items data/items.jsonl

power:
	python3 -m psba.cli power --simulate

# usage: make scaffold LANG=sw
scaffold:
	python3 -m psba.cli scaffold-language $(LANG)

clean:
	rm -rf data results runs __pycache__ psba/__pycache__ tests/__pycache__
