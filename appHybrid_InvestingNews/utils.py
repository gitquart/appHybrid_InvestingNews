import json
import os
from numpy import fabs
from selenium import webdriver
import postgresql as db
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
from nltk import tokenize


objControl=cInternalControl()
BROWSER=''
nltk.download('stopwords')
lsMyStopWords=objControl.lsMyStopWords
lsStopWord_English = set(stopwords.words('english'))
lsStopWord_Spanish= set(stopwords.words('spanish'))
lsSources=['Reuters','Investing.com','Bloomberg']
file_news='NewsDetection.txt'
file_all_words='All_words_from_all_News.txt'
lsWordAllNews=[]
#lsContentCorpus has a collection of News, the Corpus
lsContentCorpus=[]


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

def readUrl(url,page):
    try:
        returnChromeSettings()
        BROWSER.get(url)
        time.sleep(4)
        tag_article=BROWSER.find_elements_by_tag_name('article')
        no_art=len(tag_article)
        print('Total of News: ',str(no_art))
        if no_art==0:
            print('No news, shutting down...')
            os.sys.exit(0)
        #Reading articles
        for x in range(1,no_art+1):
            print(f'----------Start of Page {str(page)} New {str(x)}-------------')
            #Check Source
            lsContent=[]
            strSource=''
            txtSource=''
            time.sleep(4)
            #For source: Those from "lsSource" list have "span", the rest have "div"
            try:
                txtSource=BROWSER.find_elements_by_xpath(f'/html/body/div[5]/section/div[4]/article[{str(x)}]/div[1]/span/span[1]')[0]
            except:
                try:
                   txtSource=BROWSER.find_elements_by_xpath(f'/html/body/div[5]/section/div[4]/article[{str(x)}]/div[1]/div/span[1]')[0]
                except:
                    print(f'----------End of Page {str(page)} New {str(x)} (Most probable an ad or No content)-------------')
                    continue


            strSource=txtSource.text    
            strSource=strSource.split(' ')[1]
            linkArticle=devuelveElemento(f'/html/body/div[5]/section/div[4]/article[{str(x)}]/div[1]/a')
            BROWSER.execute_script("arguments[0].click();",linkArticle)
            if strSource in lsSources:
                #Case: Sources which news open in Investing.com
                articleContent=devuelveElemento('/html/body/div[5]/section/div[3]')
                lsContent.append(articleContent.text)
            else:
                #---To know how many windows are open----
                
                time.sleep(4)
                #Get the link with a recursive method
                linkPopUp=devuelveElementoDinamico('/html/body/div[option]/div/div/div/a',6)
                time.sleep(3)
                if linkPopUp:
                    BROWSER.execute_script("arguments[0].click();",linkPopUp)
                time.sleep(3)
                if len(BROWSER.window_handles)>1:
                    second_window=BROWSER.window_handles[1]
                    BROWSER.switch_to.window(second_window)
                    #Now in the second window
                    time.sleep(5)
                    textPage=devuelveElemento('/html/body')
                    lsContent.append(textPage.text)
                   
                    #Close Window 2
                    BROWSER.close()
                    time.sleep(4)
                    #Now in First window
                    first_window=BROWSER.window_handles[0]
                    BROWSER.switch_to.window(first_window)
                    BROWSER.refresh()
                   
            #This implementation of code is based on : 
            # https://towardsdatascience.com/using-tf-idf-to-form-descriptive-chapter-summaries-via-keyword-extraction-4e6fd857d190
            
            #START OF TF-IDF AND WORD CLOUD PROCESS
            file_New_Keywords='news_analysis\\NewAndKeywords_For_Page_'+str(page)+'_New_'+str(x)+'.txt'
            printToFile(file_New_Keywords,f'--------Start of Page {str(page)} New {str(x)} ---------------\n')
            #Pre processing
            printToFile(file_New_Keywords,f' News Content :\n')
            for content in lsContent:
                printToFile(file_New_Keywords,content+'\n')
                lsContentCorpus.append(content)
            #End of Pre procesing

            #Creating TF-IDF and its dataframe
            lsRes=getDataFrameFromTF_IDF(lsContent,file_New_Keywords)
            df=lsRes[0]
            lsFeatures=lsRes[1]
            
            
            for keywordsLimit in [20,40,60]:
                df_Sliced=df[:keywordsLimit]
                print('-------Analysis for ',str(keywordsLimit), 'keyword---------\n')
                print('Keywords limit: ',str(keywordsLimit),'\n')
                print('Features size: ',str(len(lsFeatures)),'\n')
                if keywordsLimit>len(lsFeatures):
                    print('The keywords limit is greater than the feature list')
                    os.sys.exit(0)

                printToFile(file_New_Keywords,f'-------------------First {str(keywordsLimit)} Important Keywords--------------------\n')
                printToFile(file_New_Keywords,f'-------------------Word , Tf-idf value--------------------\n')
            
                dictWord_TF_IDF={}
                for index,row in df_Sliced.iterrows():
                    line=str(row['Feature'])+' , '+str(row['tfidf_value'])
                    dictWord_TF_IDF[str(row['Feature'])]=float(str(row['tfidf_value']))
                    printToFile(file_New_Keywords,line+'\n')
                
                #Create WorldCloud from any dictionary (Ex: Word, Freq; Word, TF-IDF,....{Word, AnyValue})
                image_file='images_wordcloud\\image_page_'+str(page)+'_new_'+str(x)+'_'+str(keywordsLimit)+'_keyword.jpeg'
                createWordCloud(image_file,dictWord_TF_IDF)
                #END OF TF-IDF AND WORD CLOUD PROCESS
            
            del dictWord_TF_IDF
            del df
            del df_Sliced
            
            printToFile(file_New_Keywords,f'-------------------End of News {str(x)}--------------------\n')
            print(f'----------End of Page {str(page)} New {str(x)}-------------')
            if strSource in lsSources:
                btnCommodity= devuelveElemento('/html/body/div[5]/section/div[1]/a')
                BROWSER.execute_script("arguments[0].click();",btnCommodity)
            time.sleep(5)

        #Out of the cycle of each page    
        print(f'-End of page {str(page)}-')

        if page==3:
            print('Page 3...Starting the analysis for all Corpus')
            """
            #Print file "All words"
            printToFile(file_all_words,f'-------------------Printing All words from all news--------------------\n')
            for word in list(set(lsWordAllNews)):
                printToFile(file_all_words,f'{str(word)}\n')

            #Print file "tf idf for all news"
         
            #Creating TF-IDF and its dataframe
            file_All_News='WholeCorpus.txt'
            df_corpus=getDataFrameFromTF_IDF(lsContentCorpus,[20,40,60],file_All_News)
        
            #Generate word cloud for all documents keywords   
            """

        #query=f'update tbControl set page={str(page+1)} where id={str(objControl.idControl)}'
        #db.executeNonQuery(query)
        BROWSER.quit()


    except NameError as error:
        print(str(error))    

def pre_process_data(content):
    content = content.replace('.',' ')
    content = re.sub(r'\s+',' ',re.sub(r'[^\w \s]','',content)).lower()

    return content


def devuelveElementoDinamico(xPath,option):
    try:
        newXPath=xPath.replace('option',str(option))
        return BROWSER.find_elements_by_xpath(newXPath)[0]     
    except:
        option+=1
        devuelveElementoDinamico(xPath,option)
                   


def createWordCloud(imageName,dictWord_Weight):
    wordcloud = WordCloud().generate_from_frequencies(dictWord_Weight)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(f'{imageName}')
    del wordcloud
    
def getDataFrameFromTF_IDF(lsContent,file_test):
    
    lsFinalStopWords=[]
    #Start of "some filtering"
    #I add up the Stopwords and some cutomized Stopwords (My stop words list)
    lsFinalStopWords=list(set(lsStopWord_English) | set(lsMyStopWords) | set(lsStopWord_Spanish) )
    lsCorpus=[]
    lsVocabulary=[]
    lsVocabularyWithNoSW=[]
    data_preprocessed=pre_process_data(lsContent[0])
    lsCorpus.append(data_preprocessed)
    lsVocabulary=tokenize.word_tokenize(data_preprocessed) 
    #Remove Comple list of stop words 
    for word in lsVocabulary:
        if word not in lsFinalStopWords:
            lsVocabularyWithNoSW.append(word)
            lsWordAllNews.append(word)

    #End of "some filtering"
    

    vectorizer = TfidfVectorizer(vocabulary=list(set(lsVocabularyWithNoSW)))
            
    #fit_transform() returns
    #X sparse matrix of (n_samples, n_features)
    #Tf-idf-weighted document-term matrix.
            
    tf_idf_matrix = vectorizer.fit_transform(lsCorpus)
    lsFeatures = vectorizer.get_feature_names()
    lsDocData = tf_idf_matrix.todense().tolist()
    lsTFIDF=[]
    for tf_idf_value in lsDocData[0]:
        lsTFIDF.append(tf_idf_value)

    df = pd.DataFrame({'Feature': lsFeatures,'tfidf_value': lsTFIDF}).sort_values(by=['tfidf_value'],ascending=False)
    
    return [df,lsFeatures]
      

def printToFile(completeFileName,content):
    with open(completeFileName, 'a',encoding='utf-8') as f:
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




    