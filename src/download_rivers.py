import string, re, math, io, os, requests, datetime
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from os.path import join as pj

measure_types = {
     "gauge":"02",
     "rain":"01",
     "dam":"05",
     "all":"-1"
}

n_pages = {
     "gauge": 220,
     "rain": 278,
     "dam": 14,
     "all": 838
}

monthes = {"01":"31", "02":"28", "03":"31", "04":"30", "05":"31", "06":"30", 
           "07":"31", "08":"31", "09":"30", "10":"31", "11":"30", "12":"31"}

def get_listing_page(page_id, measure="gauge"):
    """
    """
    x = requests.get(f"http://www1.river.go.jp/cgi-bin/SrchSite.exe?KOMOKU={measure_types[measure]}&NAME=&SUIKEI=-00001&KASEN=&KEN=-1&CITY=&PAGE={page_id}")
    return x.content.decode("EUC-JP")

def parse_idx(html):
    """
    """
    return [int(x.group().split("'")[1]) for x in re.finditer("JavaScript:SiteDetail1\('[0-9]+'\)", html)]

def get_detail_page(idx):
    """
    """
    x = requests.get(f"http://www1.river.go.jp/cgi-bin/SiteInfo.exe?ID={idx}")
    return x.content.decode("EUC-JP")

def parse_details(html):
    """
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    info_table = tables[0].decode()
    search_table = tables[-1].decode()
    available_info = [int(x.group()[5]) for x in re.finditer("KIND=[0-9]&amp", search_table)]
    base = pd.read_html(info_table)[0].set_index(0)[1].to_dict()
    base["info"]=available_info
    base["coordinates"] = parse_loc(base["緯度経度"])
    return base

def clean_availability(X):
    year = X.values[2:,2:]
    nendai = [int(x[:3])*10 for x in X.values[2:,1]]
    years = []
    for i,n in enumerate(nendai):
        years.extend(map(str, (n + np.where(year[i]==year[i])[0]).tolist()))
    return years

def parse_availability(X):
    years = clean_availability(X)
    periods = []
    for year in years:
        for month in monthes:
            start = year + month + "01"
            end = year + month + monthes[month]
            periods.append((start, end))
    return periods

def parse_timestamp(data):
    timestamps=[]
    for year, time in zip(data["年月日"], data["時刻"]):
        date = [int(x) for x in year.split('/')]
        time = [int(x) for x in time.split(':')]     
        date = datetime.datetime(year=date[0], month=date[1], day=date[2], 
                                 hour=time[0]-1, minute=time[1]) + datetime.timedelta(hours=1)
        timestamps.append(date.timestamp())
    return timestamps

def parse_loc(loc):
    """
    """
    nd, nm, ns, ed, em, es = map(int, re.match("北緯 ([0-9]*)度([0-9]*)分([0-9]*)秒 東経 ([0-9]*)度([0-9]*)分([0-9]*)秒", loc).groups())
    return nd + nm/60 + ns/3600, ed + em/60 + es/3600
    
    
    
    
    
def get_dam_availability(idx):
    """
    """
    x = requests.get(f"http://www1.river.go.jp/cgi-bin/SrchDamData.exe?ID={idx}&KIND=1&PAGE=0")
    html = x.content.decode("EUC-JP")
    soup = BeautifulSoup(html, "html.parser")
    return  pd.read_html(soup.find_all("table")[-1].decode().replace('<img src="/img/ari.gif"/>', "1"))[0]

def get_dam_data_page(idx, start, end):
    """
    """
    x = requests.get(f"http://www1.river.go.jp/cgi-bin/DspDamData.exe?KIND=1&ID={idx}&BGNDATE={start}&ENDDATE={end}&KAWABOU=NO")
    return x.content.decode("EUC-JP")

def parse_dam_data(data):
    soup = BeautifulSoup(data, "html.parser")
    url = soup.find_all("a")[0].attrs["href"]
    x=requests.get(f"http://www1.river.go.jp{url}")
    content="\n".join(x.text.split("\n")[10:])
    if content:
        data = pd.read_csv(io.BytesIO(content.encode("utf-8")), header=None)
        columns = ["年月日", "時刻", "流域平均雨量", "流域平均雨量属性", "貯水量", "貯水量属性", 
                   "流入量", "流入量属性", "放流量", "放流量属性", "貯水率", "貯水率属性"]
        data.columns= columns
    else:
        data = None
    return data
    
def get_dam_data(idx, start, end):
    data = get_dam_data_page(idx, start, end)
    return parse_dam_data(data)






def get_rain_availability(idx):
    """
    """
    x = requests.get(f"http://www1.river.go.jp/cgi-bin/SrchRainData.exe?ID={idx}&KIND=1&PAGE=0")
    html = x.content.decode("EUC-JP")
    soup = BeautifulSoup(html, "html.parser")
    return  pd.read_html(soup.find_all("table")[-1].decode().replace('<img src="/img/ari.gif"/>', "1"))[0]

def get_rain_data(idx, start, end):
    data = get_rain_data_page(idx, start, end)
    return parse_rain_data(data)

def get_rain_data_page(idx, start, end):
    """
    """
    x = requests.get(f"http://www1.river.go.jp/cgi-bin/DspRainData.exe?KIND=1&ID={idx}&BGNDATE={start}&ENDDATE={end}&KAWABOU=NO")
    return x.content.decode("EUC-JP")

def parse_rain_data(data):
    soup = BeautifulSoup(data, "html.parser")
    url = soup.find_all("a")[0].attrs["href"]
    x=requests.get(f"http://www1.river.go.jp{url}")
    content="\n".join(x.text.split("\n")[10:])
    if content:
        data = pd.read_csv(io.BytesIO(content.encode("utf-8")), header=None)
        columns = ["年月日", "時刻", "雨量(mm/h)", "None"]
        data.columns= columns
    else:
        data = None
    return data






def get_gauge_availability(idx):
    """
    """
    x = requests.get(f"http://www1.river.go.jp/cgi-bin/SrchWaterData.exe?ID={idx}&KIND=1&PAGE=0")
    html = x.content.decode("EUC-JP")
    soup = BeautifulSoup(html, "html.parser")
    return  pd.read_html(soup.find_all("table")[-1].decode().replace('<img src="/img/ari.gif"/>', "1"))[0]

def get_gauge_data_page(idx, start, end):
    """
    """
    x = requests.get(f"http://www1.river.go.jp/cgi-bin/DspWaterData.exe?KIND=1&ID={idx}&BGNDATE={start}&ENDDATE={end}&KAWABOU=NO")
    return x.content.decode("EUC-JP")

def parse_gauge_data(data):
    soup = BeautifulSoup(data, "html.parser")
    url = soup.find_all("a")[0].attrs["href"]
    x=requests.get(f"http://www1.river.go.jp{url}")
    content="\n".join(x.text.split("\n")[10:])
    if content:
        columns = ["年月日", "時刻", "水位(m)",  "None"]
        data = pd.read_csv(io.BytesIO(content.encode("utf-8")), header=None, names=columns)
    else:
        data = None
    return data
    
def get_gauge_data(idx, start, end):
    data = get_gauge_data_page(idx, start, end)
    return parse_gauge_data(data)