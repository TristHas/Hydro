import requests, datetime

def frmstr(M):
    return "0" + M if len(M)==1 else M

def format_str(Y,M,D):
    Y,M,D = map(str, [Y,M,D])
    return Y, frmstr(M), frmstr(D)

def generate_dates(frm_year=2020, frm_month=1, frm_day=1, 
                   to_year=2021, to_month=1, to_day=1):
    current = datetime.datetime(year=frm_year, month=frm_month, day=frm_day)
    dates = []
    while True:
        Y,M,D = current.year, current.month, current.day
        dates.append(format_str(Y,M,D))
        current+=datetime.timedelta(days=1)
        if (Y==to_year) and (M==to_month) and (D==to_day):
            break
    return dates

def download_weather(Y,M,D,H,F):
    url = f"http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/gpv/original/{Y}/{M}/{D}/Z__C_RJTD_{Y}{M}{D}{H}0000_MSM_GPV_Rjp_Lsurf_FH{F}_grib2.bin"
    X = requests.get(url)
    if X.status_code==200:
        return X.content
    else:
        return None