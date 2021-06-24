#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 8 June 2021
@author: quart
"""
import postgresql as db
import utils as tool
from InternalControl import * 

objControl= cInternalControl()
#query=f'select app,page from tbControl where id={str(objControl.idControl)}'
#res=db.getQuery(query)
#app=res[0][0]
#page=int(res[0][1])
tool.returnChromeSettings()
url="https://www.investing.com/news/commodities-news"
tool.readUrl(url)