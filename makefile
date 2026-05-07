## EEG-MDD Classification Project Makefile

.PHONY: run train-deep clean-results clean-cache env activate rebuild

ENV_NAME=msse_277b_final_project

run:
	python -m src.run_analysis

train-deep:
	python scripts/train_deep_models.py

clean_results:
	rm -f results/tables/*.csv

clean-cache:
	rm -f data/segmented_data/*.npy
	rm -f data/segmented_data/*.csv

env: 
	conda create --name $(ENV_NAME) python=3.11 --yes
	conda run -n $(ENV_NAME) pip install -r requirements.txt 

activate:
	@echo "Run: conda activate $(ENV_NAME)"

rebuild: clean-results run 