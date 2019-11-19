import PyPDF2
import re
import pandas as pd
from rasa.nlu.model import Interpreter
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

file_name = input("Enter the name of the PDF from data folder: ")


# Train RASA model and load model
print("Loading model . . .")
interpreter = Interpreter.load('../models/nlu/tensor')

#Insert PDF
print("Parsing PDF . . .")
text = ''
pdf_reader = PyPDF2.PdfFileReader('../data/'+file_name)
for page in pdf_reader.pages:
    text += page.extractText()

cleantext = []
for k in text.split('\n'):
    cleantext.append(k)

fund_name_text = []
fund_name = []
launch_year = []
index = []
fund_name_index = []
launch_year_index = []

print("Finding values . . .")

for i in range(len(cleantext)):
    int = interpreter.parse(cleantext[i])
    if int['intent']['name'] == 'fund_launch':
        fund_name_text.append(cleantext[i])
        index.append(i)
        if len(int['entities'])>0:
            try:
                if len(int['entities'])==2:
                    if int['entities'][0]['entity'] == 'fund_name':
                        fund_name.append(int['entities'][0]['value'])
                        fund_name_index.append(i)
                        if int['entities'][1]['entity'] == 'launch_date':
                            launch_year.append(int['entities'][1]['value']) 
                            launch_year_index.append(i)
                        else:
                            launch_year.append(0)
                            launch_year_index.append(i)
            except:
                print('missed',cleantext[i])
        else:
            fund_name.append(0)
            launch_year.append(0)
            fund_name_index.append(i)
            launch_year_index.append(i)


data_text = pd.DataFrame()
data_fund = pd.DataFrame()

data_text['text'] = fund_name_text
data_text['index'] = index
data_fund['fund_name'] = fund_name
data_fund['fund_name_index'] = fund_name_index
data_fund['launch_date'] = launch_year
data_fund['launch_date_index'] = launch_year_index

data_fund[(data_fund['fund_name']!=0) & (data_fund['launch_date']!=0)].to_csv('../results/'+file_name+'.csv')

print("Done. Check file in results folder.")