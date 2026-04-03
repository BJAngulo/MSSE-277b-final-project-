environment:
	conda create --name msse_277b_final_project python=3.11 --yes
	conda run -n msse_277b_final_project pip install numpy pandas matplotlib mne scikit-learn ipykernel