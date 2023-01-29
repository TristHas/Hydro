import os
import datetime
import subprocess
import re
import configparser

class Downloader:
    def __init__(self, tar_path, bin_path):
        self.tar_path = tar_path
        self.bin_path = bin_path
        subprocess.run(['mkdir', '-p', self.tar_path])
        subprocess.run(['mkdir', '-p', self.bin_path])
    
    def set_date(self, dt):
        # URLを指定
        directory = 'http://database.rish.kyoto-u.ac.jp/arch/jmadata/data/jma-radar/synthetic/original'
        date = dt.strftime('%Y/%m/%d')
        self.timestamp = dt.strftime('%Y%m%d%H%M%S')
        self.septime = dt.strftime('%Y_%m_%d_%H_%M')
        tar_filename = 'Z__C_RJTD_'+self.timestamp+'_RDR_JMAGPV__grib2.tar'
        self.URL = directory +'/'+ date +'/'+ tar_filename
        self.tar_file_path = self.tar_path + '/' + tar_filename
        
    def get_bin_file(self):
        # binファイルがあるかを確認
        self.bin_file_path = self.bin_path+'/'+self.septime+'.bin'
        if os.path.exists(self.bin_path+'/'+self.septime+'.nc'):
            print('already exist: ',self.bin_file_path)
            self.out_available(1)
            return False
        else:
            # tarファイルを~/tarに保存
            wget_result = subprocess.getstatusoutput('wget -nc -P '+self.tar_path+' '+self.URL)  # getstatusoutputは(exitcode, output)を返す
            if wget_result[0] != 0:  # ダウンロードエラーの場合
                print(wget_result)
                error_detail = wget_result[1]
                self.out_error_URL('download', error_detail)
                return False
            else:
                # tarファイルから1kmメッシュデータのみを取り出す
                filelist = subprocess.getstatusoutput('tar -tf '+self.tar_file_path)[1].split('\n')  # tarファイル内ファイル名リスト
                ggis_name = [s for s in filelist if 'Ggis1km' in s]  # ファイル名を取得
                if ggis_name:  # リストが空でなければTrueを返す
                    ggis_name = ggis_name[0]
                    subprocess.run(['tar', '-C', self.bin_path, '-xvf', self.tar_file_path, ggis_name]) # tarファイルから/binに取り出す
                    subprocess.run(['mv', f'{self.bin_path}/{ggis_name}', f'{self.bin_file_path}'])
                    subprocess.run(['rm', self.tar_file_path])  # tarファイルは不要なので削除
                    self.out_available(1)
                    return True
                else:
                    self.out_available(0)
                    return False
                        
    def out_error_URL(self, error_type, error_detail):
        print(error_type+': '+self.URL)
        with open(self.bin_path+'/error_URL_'+self.timestamp[:4]+'.csv', 'a', encoding="utf_8_sig") as f:
            f.write(error_type+','+str(error_detail)+','+self.URL+'\n')
        self.out_available(0)
    
    def out_available(self, n):  # 1なら存在する，0なら存在しない
        with open(self.bin_path+'/available.csv', 'a', encoding="utf_8_sig") as f:
            f.write(self.timestamp+','+str(n)+'\n')