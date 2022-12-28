import pandas as pd
import csv
import os
import pickle
from collections import defaultdict
import time
import pdb
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from keras import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json
import ast
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib as mpl
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
# 저장
from joblib import Parallel, delayed
from tqdm import tqdm, tqdm_pandas
import sys
import re
from ckonlpy.tag import Twitter, Postprocessor
from konlpy.tag import Okt
import sys  
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def text_strip(text):
    text = text.strip()
    return text

def only_kor(text):

    kor = re.compile('[^ ㄱ-ㅣ가-힣+]')
    try: 
        result = kor.sub('',text)
        return result
    except: return ''


def remove_short_kakaomap_review(df):
    kakao_review = df[df.user_review_count.notnull()] # 전체 크롤링 데이터 중 카카오맵 리뷰만 추출
    remove_idx = kakao_review[kakao_review.contents.str.len()<=3].index # contents 텍스트 길이 3 이하인 리뷰 index 추출
    result = df.drop(index=remove_idx, axis=1) # 추출된 index에 해당하는 행 제거
    return result

def check(blog):
    remove_idx = []
    for i in blog.index:
        name = str(blog.name[i]).split(' ')
        title = str(blog.title[i])
        contain = False
        for word in name:
            if (word in title) or ('카페' in title):
                contain = True
        if contain == False:
            remove_idx.append(i)
    return remove_idx   

def title_cleaning(df):

    remove_idx = df.loc[df['title'].str.contains('카페')==False].index 
    result = df.drop(index=remove_idx, axis=1)
    return result

def blog_cleaning(df):
    blog = df[df.blog.notnull()] # 전체 크롤링 데이터 중 블로그 추출
    #blog = title_cleaning(blog) 
    remove_idx = check(blog)
    df = df.drop(index=remove_idx, axis=1)

    blog = df[df.blog.notnull()]
    remove_idx = blog[(blog.contents.str.len()<=50)|(blog.contents.isnull()==True)].index #내용이 없거나 50자 이내인 것들 인덱스 뽑기
    result = df.drop(index=remove_idx, axis=1)
    return result

def irrelevant_cafe(df):
    remove_idx = []
    categories = ['북카페','스터디카페,스터디룸','고양이카페',
                 '테마카페','보드카페','사주카페',
                  '만화카페','게임방,PC방','키즈카페',
                 '라이브카페','애견카페']
    for category in categories:
        tmp = df.loc[df['category']==category].index
        remove_idx.extend(tmp)
    
    result = df.drop(index=remove_idx, axis=1)
    return result

def clean(df):
    tqdm.pandas()
    #print(f'before irrelevant cafe {len(df)} data')
    df = irrelevant_cafe(df)
    #print(f'after irrelevant cafe {len(df)} data')

    df = blog_cleaning(df)
    #print(f'after short blog {len(df)} data')
    
    #df = title_cleaning(df)
    #print(f'after irrelevant title {len(df)} data')
    
    df = remove_short_kakaomap_review(df)
    #print(f'after short kakao {len(df)} data')
    df.reset_index(drop=True, inplace = True)
    
    df.contents = df.contents.apply(lambda x: only_kor(x))
    df.drop_duplicates(inplace = True)
    
    return df

def replace_word(text, token_dic):
    for i in range(len(token_dic)):
        before = str(token_dic.loc[i,'before'])
        after = str(token_dic.loc[i,'after'])
        text = text.replace(before, after)
    return text


def customization(df, token_dic) :
    df['contents'] = df['contents'].progress_apply(lambda x: replace_word(x, token_dic))
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


if __name__ == "__main__":
    num_core = os.cpu_count()
    name = sys.argv[-1]
    표제어사전 = pd.read_excel('/home/dhkim/bdm/표제어 사전.xlsx').set_index('단어')
    표제어사전 = 표제어사전.to_dict()['표제어']
    print('----reading file----')
    data= pd.read_csv(f'/home/dhkim/bdm/{name}.csv')
    data.drop_duplicates(inplace = True)
    data.reset_index(drop = True, inplace = True)
    print('----Done----')
    print(len(data))
#CLEANING--------------------------------------------------------------------------------------------------------------------
    print('----Cleaning----')
    data = clean(data)
    print(len(data))
    data.to_csv(f'/home/dhkim/bdm/{name}.csv',header = True, encoding = 'utf-8-sig',index = False)
    token_dic = pd.read_csv(f'/home/dhkim/bdm/token_dic.csv')

#CUSOMIZING--------------------------------------------------------------------------------------------------------------
    print('----Customizing----')
    data_chunks = np.array_split(data, num_core)
    with Parallel(n_jobs = num_core, backend="multiprocessing") as parallel:
        results = parallel(delayed(customization)(data_chunks[i], token_dic) for i in range(num_core))
    for i,data in enumerate(results):
        if i == 0:
            result = data
        else:
            result = pd.concat([result, data], axis = 0)
    
    result.to_csv(f'/home/dhkim/bdm/{name}.csv',header = True, encoding = 'utf-8-sig',index = False)
