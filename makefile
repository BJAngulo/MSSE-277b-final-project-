environment:
	conda create --name msse_277b_final_project python=3.11 --yes
	conda run -n msse_277b_final_project python -m pip install "numpy<2.0" pandas matplotlib mne scikit-learn ipykernel torch
	