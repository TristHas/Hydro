from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA 
from sklearn import preprocessing
import numpy as np

class Linear(LinearRegression):
    def __init__(self):
        self.sscaler = preprocessing.StandardScaler()
        super().__init__()
        
    def fit(self,x,y):
        x = self._preprocess(x)
        super().fit(x,y)
        
    def predict(self,x):
        x = self.sscaler.transform(x)
        x = self.pca.transform(x)
        return super().predict(x)

        
    def _preprocess(self,x):
        #正規化
        self.sscaler.fit(x)
        x = self.sscaler.transform(x) 
        
        #累積寄与率が0.9を超えるような次元数を求める
        pca = PCA(n_components=x.shape[1]-1)
        _ = pca.fit(x)
        cumcon = np.cumsum(pca.explained_variance_ratio_)
        
        #次元削減
        self.pca = PCA(n_components=np.min(np.where(cumcon>0.9)))
        return self.pca.fit_transform(x)
        