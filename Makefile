.PHONY: setup train demo
setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

train:
	python src/train_cls.py --data data/sample --epochs 1 --batch 8

demo:
	streamlit run demo/streamlit_app.py
