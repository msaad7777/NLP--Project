# NLP--Project
There are few ways to execute the code:

1) 
pip install -r requirements.txt

python text_classifier.py

2) 
.\venv\Scripts\activate

cd in venv 

then 

pip install -r requirements.txt

python text_classifier.py

3) Playbook naive_bayes.ipnb, to execute each cell.


Code Structure :
1) input folder is were we have Excel sheet , our actual data
2) Output Folder is were all our output will be saved, vectorizers transformers etc., will save in picke files
3) Source will have model, procesing and utils.py files. 
4) config.py will have model configurtation - random speeds ,etc.,
5) After some attempts we decided to have all the code in a single file as its easier to run all the code at once.
So we have final code in text_classifer.py