import cassandraUtil as db
import json
import os
from selenium import webdriver
import chromedriver_autoinstaller
import uuid
import time
from InternalControl import cInternalControl
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
from nltk.corpus import stopwords
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt


objControl=cInternalControl()
BROWSER=''
nltk.download('stopwords')
st = set(stopwords.words('english'))


def returnChromeSettings():
    global BROWSER
    chromedriver_autoinstaller.install()
    options = Options()
    ua = UserAgent()
    userAgent = ua.random
    options.add_argument("start-maximized")
    options.add_argument(f"user-agent={userAgent}")
    options.add_argument("--no-sandbox")

    if objControl.heroku:
        #Chrome configuration for heroku
        options.binary_location=os.environ.get("GOOGLE_CHROME_BIN")
        options.add_argument("--disable-dev-shm-usage")
        BROWSER=webdriver.Chrome(executable_path=os.environ.get("CHROMEDRIVER_PATH"),chrome_options=options)
    else:
        BROWSER=webdriver.Chrome(options=options)  



"""
readUrl

Reads the url from the jury web site
"""

def readUrl():
    try:
        returnChromeSettings()
        url="https://www.investing.com/news/commodities-news"
        BROWSER.get(url)
        #print('Waiting for banner to appear')
        #time.sleep(5)
        #btnCloseAlert=devuelveElemento('/html/body/div[6]/div[2]/i')
        #BROWSER.execute_script("arguments[0].click();",btnCloseAlert)
        #Reading articles
        #sw = set(stopwords.words('english'))
        for x in range(1,38):
            print(f'----------Start of New {str(x)}-------------')
            linkArticle=devuelveElemento(f'/html/body/div[5]/section/div[4]/article[{str(x)}]/div[1]/a')
            BROWSER.execute_script("arguments[0].click();",linkArticle)
            articleContent=devuelveElemento('/html/body/div[5]/section/div[3]')
            strContent=articleContent.text
            #This implementation of code is based on : 
            # https://towardsdatascience.com/using-tf-idf-to-form-descriptive-chapter-summaries-via-keyword-extraction-4e6fd857d190
            #Pre processing
            file_test='Results.txt'
            printToFile(file_test,f'--------Start of New {str(x)} ---------------\n')
            printToFile(file_test,f' News Content :\n')
            printToFile(file_test,strContent+'\n')
            strContent = strContent.replace('.',' ')
            strContent = re.sub(r'\s+',' ',re.sub(r'[^\w \s]','',strContent) ).lower()
            lsCorpus=[]
            lsCorpus.append(strContent)
            #Start of getting keywords
            vectorizer = TfidfVectorizer(stop_words=st)
            """
            fit_transform() returns
            X sparse matrix of (n_samples, n_features)
            Tf-idf-weighted document-term matrix.
            """
            tf_idf_matrix = vectorizer.fit_transform(lsCorpus)
            lsNames = vectorizer.get_feature_names()
            lsDocData = tf_idf_matrix.todense().tolist()
            lsTFIDF=[]
            for tf_idf_value in lsDocData[0]:
                lsTFIDF.append(tf_idf_value)
            keywordsLimit=10
            printToFile(file_test,f'-------------------First {str(keywordsLimit)} Important Keywords--------------------\n')
            printToFile(file_test,f'-------------------Word , Tf-idf value--------------------\n')
            # Create a dataframe with the results
            dictWord_TF_IDF={}
            df = pd.DataFrame({'Feature': lsNames,'tfidf_value': lsTFIDF}).sort_values(by=['tfidf_value'],ascending=False)[0:keywordsLimit]
            for index,row in df.iterrows():
                line=str(row['Feature'])+' , '+str(row['tfidf_value'])
                dictWord_TF_IDF[str(row['Feature'])]=float(str(row['tfidf_value']))
                printToFile(file_test,line+'\n')
                
            #Create WorldCloud from any dictionary (Ex: Word, Freq; Word, TF-IDF,....{Word, AnyValue})
            wordcloud = WordCloud().generate_from_frequencies(dictWord_TF_IDF)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()

            #End of getting keywords
            printToFile(file_test,f'-------------------End of News {str(x)}--------------------\n')
            print(f'----------End of New {str(x)}-------------')
            btnCommodity= devuelveElemento('/html/body/div[5]/section/div[1]/a')
            BROWSER.execute_script("arguments[0].click();",btnCommodity)
            time.sleep(5)
        
    except NameError as error:
        print(str(error))    

    
    
      

def printToFile(completeFileName,content):
    with open(completeFileName, 'a') as f:
        f.write(content)
    f.close()    
 
                                       
def devuelveJSON(jsonFile):
    with open(jsonFile) as json_file:
        jsonObj = json.load(json_file)
    
    return jsonObj 

def devuelveElemento(xPath):
    cEle=0
    while (cEle==0):
        cEle=len(BROWSER.find_elements_by_xpath(xPath))
        if cEle>0:
            ele=BROWSER.find_elements_by_xpath(xPath)[0]

    return ele  

def devuelveListaElementos(xPath):
    cEle=0
    while (cEle==0):
        cEle=len(BROWSER.find_elements_by_xpath(xPath))
        if cEle>0:
            ele=BROWSER.find_elements_by_xpath(xPath)

    return ele     
    

    