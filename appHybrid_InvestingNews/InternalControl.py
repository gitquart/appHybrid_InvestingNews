import os

class cInternalControl(object):
    idControl=12
    timeout=70
    hfolder='appHybrid_InvestingNews' 
    heroku=False
    rutaHeroku='/app/'+hfolder+'/'
    rutaLocal=os.getcwd()+'\\'+hfolder+'\\'
    download_dir=''
      