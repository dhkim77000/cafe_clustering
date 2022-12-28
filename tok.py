import pandas as pd
import csv
import os
import pickle
from collections import defaultdict
from konlpy.tag import Kkma
import time
import pdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras import models
from tensorflow.keras.layers import Embedding, Dense, LSTM, GRU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import ast
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
import matplotlib.font_manager as fm
import matplotlib as mpl
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# 저장
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_pandas
import sys
import re
from ckonlpy.tag import Twitter, Postprocessor
import sys  







def tokenization(df, token_dic):
    tqdm.pandas()
    twitter = Twitter()
    done = set()
    for i in range(len(token_dic)):
        word = token_dic.loc[i,'after']
        if word not in done:
            done.add(word)
            word = str(word)
            if word != 'nan':twitter.add_dictionary(word, token_dic.loc[i,'type']) 

    df['contents'] = df['contents'].progress_apply(lambda x: twitter.pos(str(x), norm=True, stem=True))
    df['noun'] = df['contents'].progress_apply(lambda x: [w[0] for w in x if w[1]=='Noun']) 
    df['adj'] = df['contents'].progress_apply(lambda x: [w[0] for w in x if w[1]=='Adjective']) 
    return df

def 표제어처리(표제어사전, words_to_be_changed, text):

    if type(text) == 'str': text = ast.literal_eval(text) 
    result = []  
    for data in text:
        word,w_type = data 
        if word in words_to_be_changed: 
            replace = 표제어사전[word]
            if replace != 'n': 
                result.append((표제어사전[word], w_type)) 
        else:
            result.append((word, w_type))
    return result

def init_jvm(jvmpath=None):
    if jpype.isJVMStarted():
        return
    jpype.startJVM(jpype.getDefaultJVMPath())

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category= UserWarning) 
    num_core = os.cpu_count()
    name = sys.argv[-1]
    표제어사전 = pd.read_excel('/home/dhkim/bdm/표제어 사전.xlsx').set_index('단어')
    표제어사전 = 표제어사전.to_dict()['표제어']

    print('----reading file----')
    data = pd.read_csv(f'/home/dhkim/bdm/{name}.csv')
    token_dic = pd.read_csv(f'/home/dhkim/bdm/token_dic.csv')
    data.drop_duplicates(inplace = True)
    data.reset_index(drop = True, inplace = True)
    print('----Done----')
#TOKENIZATION-----------------------------------------------------------------------------------------------------------
    print('----Tokenizing----')
    data_chunks = np.array_split(data, num_core)
    print(f"Parallelizing with {num_core} cpus")
    with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
        results = parallel(delayed(tokenization)(data_chunks[i], token_dic) for i in range(num_core))
    for i,data in enumerate(results):
        if i == 0:
            result = data
        else:
            result = pd.concat([result, data], axis = 0)
    try:
        result.drop(['blog', 'link'], axis = 0, inplace = True)
    except: 1
    
    result.to_csv(f'/home/dhkim/bdm/tokened.csv', header = True, index = False, encoding = 'utf-8-sig') 