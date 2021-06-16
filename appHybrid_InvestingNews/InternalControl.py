import os

class cInternalControl(object):
    idControl=1
    timeout=70
    hfolder='appHybrid_InvestingNews' 
    heroku=False
    rutaHeroku='/app/'+hfolder+'/'
    rutaLocal=os.getcwd()+'\\'+hfolder+'\\'
    download_dir=''
    lsMyStopWords=['reuters','by','com','u','s','have','has','said','the','are','his','her','would','say','marketwatch',
                   'since','could','newsletters','nwe2050','nowbrokerstoolseconomic','comworldamericasperuscastilloleadselectionwith501votesafterallballotstallied20210615',
                   'etfsnysesiljsilxsinv','oilprice','ev','gm']
      