import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from matplotlib.colors import ListedColormap

data = pd.read_csv("C:/Users/gokay/OneDrive/Masaüstü/DerinOgrenme_1/DerinOgrenme_Dersler/Derin Ogrenme 5.2/1/breastcancer.csv")
data.head()
data.drop(["Unnamed: 32","id"],inplace=True,axis=1)


data = data.rename(columns={"diagnosis":"target"})       
data.head(13)

sns.countplot(x ="target",data=data)
print()

data["target"] = data["target"].replace({"M":1,"B":0})

print("Data Len : ",len(data))
describe = data.describe()
print("Describe",describe)
print("Data Shape : ",data.shape)
print("Data İnfo : ",data.info)
#%%
corr_matrix = data.corr() 

plt.figure()
sns.clustermap(corr_matrix,annot=True,fmt ="0.2f",figsize=(18,16))
# annot = True  sayıların gözükmesini sağlar
# figsize boyutlandırma
# fmt 0dan sonra 2 rakam olsun 0.21 0.33 gibi
plt.title("Korelasyon")
plt.show()


deger = 0.75
filtre = np.abs(corr_matrix["target"]) > deger # abs mutlak değere alır
corr_features = corr_matrix.columns[filtre].tolist() # 0.75 değerinden büyük olanları listeye alır
sns.clustermap(data[corr_features].corr(),annot=True,fmt="0.2f",figsize=(16,14))
plt.title("Koreslasyon Değeri 0.75 Değerinden Büyük Olanlar")
plt.show()

#boxplot
# fonksiyonu, bir DataFrame'i uzun formata dönüştürmek için kullanılır. Yani, veri çerçevenizdeki birden fazla sütunu 
# (özellikleri) tek bir sütunda birleştirip, bu sütunun altında her bir özelliğe karşılık gelen değerleri listelemenizi sağlar.
data_melted = pd.melt(data,id_vars="target",
                      var_name="features",
                      value_name="values")
plt.figure()
sns.boxplot(x="features",y="values",hue="target",data=data_melted)
# hue = target anlamı target değişkenine göre renklendirme yapar
plt.xticks(rotation=90)
plt.show()

#   """standartization yapmamız lazım doğru gözükmesi için ( ileride yapıcaz ) """

# pairplot
sns.pairplot(data[corr_features],diag_kind="kde",markers="+",hue="target")
plt.show()
#burda kuyruk sağ tarafa doğru uzadıysa pozitif skewnes   sola doğru uzadıysa negatif skewnes
#%% Aykırı Değer Tespiti (Anomaly Detection) işlemi gerçekleştiriyor.

from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor

y = data.target
x = data.drop(["target"],axis=1)
columns = x.columns.tolist()

clf = LocalOutlierFactor()
y_pred = clf.fit_predict(x)
print(y_pred)
# LOF skoru > 1 olan bir veri noktası, aykırı olma olasılığı yüksek bir noktadır ve bu nedenle 
# fit_predict çıktısında -1 olarak işaretlenir (yani aykırı değer olarak kabul edilir).
# LOF skoru < 1 olan bir veri noktası, normal kabul edilir ve fit_predict çıktısında 1 olarak işaretlenir (yani normal değer olarak kabul edilir).

x_score = clf.negative_outlier_factor_
# clf isimli bir modelin (örneğin, bir Local Outlier Factor (LOF) modeli) negatif dışlayıcı faktörlerini (negative_outlier_factor_   (-1)) alır 
# ve bunları x_score değişkenine atar.
outlier_score = pd.DataFrame()
outlier_score["score"] = x_score

#threshold
deger = -2.5
filtre = outlier_score["score"] < deger
outlier_index = outlier_score[filtre].index.tolist()

plt.figure()
plt.scatter(x.iloc[outlier_index,0],x.iloc[outlier_index,1],color="blue",s=50,label="Outlier Filtre")

plt.scatter(x.iloc[:,0],x.iloc[:,1],color="k",s=3,label="Data Points")
radius = (x_score.max()-x_score)/(x_score.max()-x_score.min())
outlier_score["radius"]=radius
plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors="r",facecolors="none",label="Outlier Scores")
plt.legend()
plt.show()
# aykırı değerler mavi renkte , veriler siyah renkle
# Kırmızı halkanın amacı, aykırı değerlerin (outlier) görselleştirilmesi ve bu değerlerin ne kadar "şiddetli" veya "belirgin" olduklarını vurgulamaktır


x = x.drop(outlier_index)
# x veri çerçevesinden, outlier_index listesinde belirtilen indekslere sahip olan satırları kaldırır.
# Bu işlem, aykırı değerlerin (outlier) model üzerindeki olumsuz etkilerini azaltmak ve daha sağlıklı bir veri seti elde etmek için yapılır.
y = y.drop(outlier_index).values
# y, hedef değişkeni temsil eder (bu durumda kanser durumu olan target değişkeni).
# Bu satır, y dizisinden de outlier_index listesinde belirtilen aykırı değerlerin indekslerine sahip olan satırları kaldırır.

#%%
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#%% standardscaler
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# xtrain ile eğiticeğim için  xtesti fit etmek yerine sadece transform yapıyorum
# böylece xtrain ile eğitilmiş verimi x_Test üzerinde uyguluyorum ve scale etmiş oluyorum

X_train_df = pd.DataFrame(X_train,columns=columns)
X_train_dfdescribe = X_train_df.describe()
X_train_df["target"] = Y_train

#boxxplot

data_melted = pd.melt(X_train_df,id_vars="target",
                      var_name="features",
                      value_name="value")

plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)
# hue = target anlamı target değişkenine göre renklendirme yapar
# data=data_melted   Kutunun verisinin data_melted veri çerçevesinden alındığını belirtir.
plt.xticks(rotation=90)
plt.show()

# pair plot
# scewnes 
sns.pairplot(X_train_df[corr_features],markers="+",hue="target",diag_kind="kde")
plt.show()

#%% basic knn methot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# VERİMİZİ STANDARTLAŞTIRDIK ARTIK EĞİTİME HAZIRLADIK
# outlier varsa KNN güze çalışmaz  - büyük veride sıkıntılı - çok fazla feature varsa sıkıntı çıkabilir

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_test, Y_test)
print("Score : ",score)
print("acc : ",acc)
print("cm : ",cm)
#   107    1  ben 108tane iyi huyludan 107 tanesini doğru tahmin etmişim 1 tanesi yanlış
#    7    56  ben 63tane kötü huyludan 56 tanesini doğru tahmin etmişim 7 tanesi yanlış

#%%  choose best parameters
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import accuracy_score, confusion_matrix

def KNN_best_parameters(x_train,x_test,y_train,y_test):
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors=k_range,weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid,cv=10,scoring="accuracy")
    grid.fit(x_train,y_train)
    print("Best training score {}  witth parameters {} ".format(grid.best_score_,grid.best_params_))   
    print()
    # en iyi parametrelere sahip olduğum için  ben grid kullanabilirim neyde kullanırım test veri setinde 
    # kullanmanın 2 yolu var şimdi 1. yolu kullanıcaz ama bu tercih etmediğim bir yol

    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train,y_train)
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)
    
    cm_test = confusion_matrix(y_test, y_pred_test)
    cm_train = confusion_matrix(y_train, y_pred_train)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    
    print("Test Score : {} , Train Score : {} ".format(acc_test,acc_train))
    print()
    print("CM Test Score : {} ".format(cm_test))
    print()
    print("CM Train Score : {} ".format(cm_train))
    return grid

grid = KNN_best_parameters(X_train,X_test,Y_train,Y_test)
    # TRAİN SKOR  TEST SKORUNDAN FAZLA ÇIKIYORSA BURDA BİR OVERFİTTİNG SÖZ KONUSUDUR 

#%%   PCA
from sklearn.decomposition import PCA

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components=2)
pca.fit(x_scaled)
pca.transform(x_scaled)

X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca,columns=["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x="p1",y="p2",hue="target",data=pca_data)
plt.title("PCA : P1 VS P2")

X_train_pca,X_test_pca,Y_train_pca,Y_test_pca = train_test_split(X_reduced_pca,y,test_size=0.3,random_state=42)

grid_pca = KNN_best_parameters(X_train_pca,X_test_pca,Y_train_pca,Y_test_pca)


cmap_light = ListedColormap(["orange","cornflowerblue"])
cmap_bold = ListedColormap(["darkorange","darkblue"])
h = .05
X = X_reduced_pca
x_min,x_max = X[:,0].min() -1 ,X[:,0].max() +1
y_min,y_max = X[:,0].min() -1 ,X[:,0].max() +1


xx ,yy = np.meshgrid(np.arange(x_min,x_max,h),
                     np.arange(y_min,y_max,h))

Z = grid_pca.predict(np.c_[xx.ravel(),yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)

plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,
            edgecolors="k",s=20)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("%i-Class classification (k = %i , weights = ''%s')"
          % (len(np.unique(y)),grid_pca.best_estimator_.n_neighbors,grid_pca.best_estimator_.weights))


#%% NCA

nca = NeighborhoodComponentsAnalysis(n_components=2,random_state=42)
nca.fit(x_scaled,y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca,columns=["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x="p1",y="p2",hue="target",data=nca_data)
plt.title("NCA : P1 VS P2")


X_train_nca,X_test_nca,Y_train_nca,Y_test_nca = train_test_split(X_reduced_nca,y,test_size=0.3,random_state=42)
grid_nca = KNN_best_parameters(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)


cmap_light = ListedColormap(["orange","cornflowerblue"])
cmap_bold = ListedColormap(["darkorange","darkblue"])

h = .2
X = X_reduced_nca
x_min,x_max = X[:,0].min() -1 , X[:,0].max() +1
y_min,y_max = X[:,0].min() -1 , X[:,0].max() +1

xx,yy = np.meshgrid(np.arange(x_min,x_max,h),
                    np.arange(y_min,y_max,h))

Z = grid_nca.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure()
plt.pcolormesh(xx,yy,Z,cmap=cmap_light)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_bold,
            edgecolors="k",s=20)
plt.xlim(xx.min(),xx.max())
plt.ylim(yy.min(),yy.max())
plt.title("%i-Class classification (k = %i , weights = ''%s')"
          % (len(np.unique(y)),grid_nca.best_estimator_.n_neighbors,grid_nca.best_estimator_.weights))
plt.show()
