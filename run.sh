mkdir -p ./checkpoint/textsim
mkdir -p ./checkpoint/predict_num
conda create -n goodwang python=3.8.10
conda activate goodwang
pip install ark-nlp
pip install scikit-learn 
pip install pandas
pip install elasticsearch
pip install openpyxl
pip install python-Levenshtein
python data_process.py
python textsim.py
python predictnum.py
python predict.py
