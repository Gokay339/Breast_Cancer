import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler
# StandartScaler : Verileri normalleştirerek ortalama 0  , standart sapması 1 olucak şekilde yapar
from sklearn.model_selection import train_test_split, GridSearchCV
# train_test_split : Test ve eğitim verilerini yapmamızı sağlar
# GridSearchCV : KNN ile ilgili best prametleri seçerken kullanıcam
from sklearn.metrics import accuracy_score, confusion_matrix
# accuracy_score :  Modelin doğru tahmin oranını ölçmek için kullanılır
# confusion_matrix : Modelin performansını sadece doğruluk (accuracy) ile değerlendirmek bazen yanıltıcı olabilir.
# confusion_matrix ile sınıflandırmanın hangi sınıflarda doğru ya da yanlış olduğunu detaylı şekilde görebilirsiniz.
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
# KNN : En yakın komşu algoritması
# NeighborhoodComponentsAnalysis : En iyi sınıflandırma performansı sağlayacak bir dönüşüm gerçekleştirir
# LocalOutlierFactor : Verideki anormal veya aykırı değerleri (outliers) tespit eder.
from sklearn.decomposition import PCA
# PCA : Verideki çok sayıda özelliği, önemli bilgileri koruyarak daha az sayıda özelliğe indirger.
# PCA, verinin çok karmaşık olduğu durumlarda, tüm özellikleri kullanmak yerine, veriyi özetler ve
# önemli olan kısımlarını bulur. Böylece veri, daha az boyutlu hale gelir ama önemli bilgiler kaybolmaz.
# Örneğin, bir veri setinde 10 özellik varsa, bu 10 özelliğin yerine 2 ya da 3 ana özelliğe indirgenebilir.


import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("C:/Users/gokay/OneDrive/Masaüstü/DerinOgrenme_1/DerinOgrenme_Dersler/Derin Ogrenme 5.2/1/breastcancer.csv")
data.head(5)
data.drop(["Unnamed: 32", "id"], inplace=True, axis=1)
# Kodumuzdan id ve Unnamed32 diye 2 sütun vardı onları çıkattık
# inplace=False ile değişiklik veri kopyasında yapılır ve orijinal veri etkilenmez.
# inplace=True ile veri orijinalde değişir.
# axis = 1  Sütunları etkile , axis = 0 satırları etkile demek

data = data.rename(columns={"diagnosis":"target"})
# Diagnosis kolonun ismini değiştirdik ve target yaptık


sns.countplot(data["target"])  # target sütununu grafiğe çevirdik
print(data.target.value_counts())
# 357 Tane B yani iyi huylu - 212 Tane M yani kötü huylu tümör var

data["target"] = data["target"].replace({"M":1,"B":0})
# M VE B Değerleri String bunları kaldırmamız gerekiyor
# M Kötü huylu tümör değeri 1 oldu , B İyi huylu tümör değeri 0 oldu

print(len(data))
print("Data Shape : ",data.shape)
data.info()
describe = data.describe()

#%%


corr_matrix = data.corr()  # Korelasyonlarına bakıyoruz 
# Korelasyon veri tablosunda kategorical yani string değerler gözükmez 
# ama onehot encoding veya labelencoding kullanarak göstermesini sağlayabiliriz


sns.clustermap(corr_matrix,annot=True,fmt ="0.2f")
# annot = True  sayıların gözükmesini sağlar
# figsize boyutlandırma
# fmt 0dan sonra 2 rakam olsun 0.21 0.33 gibi
plt.title("Korelasyon")
plt.show()


# corr değerlerinde sadece 0.75 üzerindekileri gösterme
deger = 0.75
filtre = np.abs(corr_matrix["target"]) > deger      #np.abs mutlak değere alır böylece negatif çıkmaz
corr_features = corr_matrix.columns[filtre].tolist()
# 0.75ten büyük olan değerleri listeye çevir
sns.clustermap(data[corr_features].corr(),annot=True,fmt ="0.2f")
plt.title("Korelasyon 0.75den büyük olanlar")
plt.show()



#boxplot
data_melted = pd.melt(data,id_vars="target", # kimlik değişkeni" olarak tanımlanır ve bu sütun dönüştürme sırasında sabit kalır.
                      var_name="features",   # Dönüşüm sonucu oluşacak olan yeni sütunun adı features olacaktır. Bu sütun, orijinal veri çerçevesindeki diğer sütun adlarını içerecek.
                      value_name="value")    # Diğer sütunlardaki değerlerin yer alacağı yeni sütunun adı value olacaktır.
# pd.melt()  geniş formatlı bir veri setini uzun formata dönüştürmek için kullanılır.
 #data: Dönüştürmek istediğiniz DataFrame.
#id_vars: Sabit kalmasını istediğiniz sütunlar. Bu örnekte target sabit kalacak.
#var_name: Yeni oluşturulacak sütunun adı. Bu örnekte "features" olarak adlandırılıyor.
#value_name: Diğer sütunlardaki değerlerin bulunduğu yeni sütunun adı. Bu örnekte "value" olarak adlandırılıyor.


plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)
# hue = target anlamı target değişkenine göre renklendirme yapar
# data=data_melted   Kutunun verisinin data_melted veri çerçevesinden alındığını belirtir.
plt.xticks(rotation=90) # dik yapmaya yarar
plt.show()

#   """standartization yapmamız lazım doğru gözükmesi için"""

# pairplot
sns.pairplot(data[corr_features],diag_kind="kde",markers="+",hue="target")
plt.show()
#burda kuyruk sağ tarafa doğru uzadıysa pozitif skewnes   sola doğru uzadıysa negatif skewnes



#%% outlier bulma

# outlier modelin içindeki aykırı değerlerdir 
# Aykırı değerler, bir veri setinde diğer gözlemlerden farklı olan veya oldukça uzak duran veri noktalarıdır. 
# outlier ayıklanmazsa modeli yanlış yönlendirebilir
# outlier detection yaparken kullanıcağımız sistem --- Density based ODS bu yöntemin içinden (local outiler factory LOF) kullanıcaz
#                                                      LOF yöntemini kullanıcaz çünkü bizim verilerimiz secure data

# compare local density of one point to     local density of its KNN
#  Bir noktanın yerel yoğunluğunu, o noktanın K-en yakın komşularının (KNN) yerel yoğunluğu ile karşılaştır.

# Yerel Yoğunluk (Local Density): LOF, bir veri noktasının komşularına olan uzaklıklarına bakarak o noktanın yoğunluğunu hesaplar. 
# Yoğunluk hesaplanırken genellikle K-en yakın komşu (K-Nearest Neighbors, KNN) yöntemi kullanılır.

# Yerel Yoğunluk Karşılaştırması: LOF, bir noktayı aykırı değer olarak değerlendirmek için o noktanın yerel yoğunluğunu, 
# çevresindeki komşularının yerel yoğunluğu ile karşılaştırır.
# Eğer bir noktanın yerel yoğunluğu komşularının yerel yoğunluğuna göre çok daha düşükse, bu nokta aykırı olarak değerlendirilir.


#LOF Nasıl Çalışır?
# KNN (K-en Yakın Komşu) Seçimi: Her bir veri noktası için K-en yakın komşular (genellikle belirlenen bir K değeri) bulunur. 
# K-en yakın komşular, veri noktasına en yakın olan K adet veri noktasıdır.

# Ulaşım Mesafesi (Reachability Distance): LOF algoritması, bir noktanın komşularına olan mesafesini belirlemek için "ulaşılabilirlik mesafesi" adı verilen bir kavramı kullanır.

# Yerel Erişilebilirlik Yoğunluğu (Local Reachability Density, LRD): Her bir veri noktası için yerel erişilebilirlik yoğunluğu hesaplanır.
# Bu yoğunluk, o noktanın komşularına olan ortalama uzaklığının tersi olarak ifade edilir. Daha küçük uzaklıklar, daha yüksek yoğunluk anlamına gelir.

# LOF Skoru: LOF algoritması, her veri noktası için LOF skoru hesaplar. Bu skor, noktanın yerel yoğunluğunu komşularının yerel yoğunluğuna kıyasla değerlendirir.
# LOF skoru şu şekilde yorumlanır:
# LOF Skoru ≈ 1: Veri noktası, komşularıyla benzer yoğunluğa sahiptir ve aykırı değer olarak değerlendirilmez.
# LOF Skoru > 1: Veri noktası komşularına göre daha düşük yoğunluğa sahiptir ve aykırı olma olasılığı yüksektir.
# LOF Skoru < 1: Veri noktası, komşularına göre daha yüksek yoğunluğa sahiptir ve aykırı olmadığı düşünülür.


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
outlier_score["radius"] = radius
plt.scatter(x.iloc[:,0],x.iloc[:,1],s=1000*radius,edgecolors="r",facecolors="none",label="Outlier Scores")
plt.legend()
plt.show()
# aykırı değerler mavi renkte , veriler siyah renkle
# Kırmızı halkanın amacı, aykırı değerlerin (outlier) görselleştirilmesi ve bu değerlerin ne kadar "şiddetli" veya "belirgin" olduklarını vurgulamaktır



# drop outliers


x = x.drop(outlier_index)
y = y.drop(outlier_index).values





#%% train test split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#%% standardscaler

# listemiz var [1,2,3,4,5] mean=3  std=1.5
#                 x değerleri

#  (x - mean) / std = ilk adımda [-2,-1,0,1,2] sonra -2/1.5 , -1/1.5 



sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# xtrain ile eğiticeğim için  xtesti fit etmek yerine sadece transform yapıyorum
# böylece xtrain ile eğitilmiş verimi x_Test üzerinde uyguluyorum ve scale etmiş oluyorum

X_train_df = pd.DataFrame(X_train,columns=columns)
X_train_dfdescribe = X_train_df.describe()
X_train_df["target"] = Y_train

#boxx plot
data_melted = pd.melt(X_train_df,id_vars="target", 
                      var_name="features",   
                      value_name="value")  

plt.figure()
sns.boxplot(x="features",y="value",hue="target",data=data_melted)
# hue = target anlamı target değişkenine göre renklendirme yapar
# data=data_melted   Kutunun verisinin data_melted veri çerçevesinden alındığını belirtir.
plt.xticks(rotation=90) # dik yapmaya yarar
plt.show()

# pair plot
# scewnes 
sns.pairplot(X_train_df[corr_features],diag_kind="kde",markers="+",hue="target")
plt.show()

#%% basic knn methot

# VERİMİZİ STANDARTLAŞTIRDIK ARTIK EĞİTİME HAZIRLADIK
# outlier varsa KNN güze çalışmaz  - büyük veride sıkıntılı - çok fazla feature varsa sıkıntı çıkabilir

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train,Y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
acc = accuracy_score(Y_test, y_pred)
score = knn.score(X_test,Y_test)
print("Score : ",score)
print("acc : ",acc)
print("cm : ",cm)
#   107    1  ben 108tane iyi huyludan 107 tanesini doğru tahmin etmişim 1 tanesi yanlış
#    7    56  ben 63tane kötü huyludan 56 tanesini doğru tahmin etmişim 7 tanesi yanlış

#%%  choose best parameters

def KNN_best_parameters(x_train,x_test,y_train,y_test):
    k_range = list(range(1,31))
    weight_options = ["uniform","distance"]
    print()
    param_grid = dict(n_neighbors = k_range,weights = weight_options)
    
    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid,cv=10,scoring="accuracy") # CV AMACI OVERFİTTİNG AZALTMAK
    grid.fit(x_train,y_train)
    print("Best training score {}  witth parameters {} ".format(grid.best_score_,grid.best_params_))   
    print()
    # en iyi parametrelere sahip olduğum için  ben grid kullanabilirim neyde kullanırım test veri setinde 
    # kullanmanın 2 yolu var şimdi 1. yolu kullanıcaz ama bu tercih etmediğim bir yol
    
    knn = KNeighborsClassifier(**grid.best_params_)
    knn.fit(x_train,y_train)
    
    y_pred_test = knn.predict(x_test)
    y_pred_train = knn.predict(x_train)   # overfitting var mı kontrol etmek için train verisetimede bakıyorum
    
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
grid =KNN_best_parameters(X_train,X_test,Y_train,Y_test)

# TRAİN SKOR  TEST SKORUNDAN FAZLA ÇIKIYORSA BURDA BİR OVERFİTTİNG SÖZ KONUSUDUR 
    
    
#%%   PCA

# PCA yani Temel Bileşen Analizi, veri boyutunu azaltmak ve verinin en önemli özelliklerini çıkarmak için kullanılan
# bir istatistiksel teknik ve makine öğrenimi algoritmasıdır. 
# PCA'nin temel amacı, boyut indirgeme ve veri sıkıştırmadır, ancak bunu yaparken verinin en önemli özelliklerini kaybetmeden korumayı hedefler.
# Mesela elimizde 30 boyutlu veri varsa bunu görselleştiremeyiz bu nedenle PCA kullanarak 2 boyuta düşürücez

"""
cov(x,y) = [var(x) cov(x,y)]    cov(x,y) = cov(y,x)  nedeni cov(x,y) = E[xy] - E[X] X E[Y] bu nedenle = cox(y,x)
           [cov(y,x) var(y)]
"""

# Eigenvector: Bir matrisle çarpıldığında yönü (direction) değişmeyen vektördür. Bu nedenle, özvektörler, matrisin uyguladığı 
# dönüşümde hangi yönlerin sabit kaldığını gösterir. 
# Önce formül ile verimin ortalamasını buluyorum
                                                                        #                         _   x  = x ekseni   y = y  ekseni
#                                                 x̄ = x eksenindeki ortalama                      y  = y eksenindeki ortalama 
#  Formül     x - x̄  = x
# kendilerine eşitliyorum böylece  ortalamalarını çıkarttığım için veriyi 0 merkezli hale getiriyorum artık verinin merkezi  0.0 noktası


# Eigenvalue: Bir özvektör matrisle çarpıldığında, vektörün büyüklüğünün (magnitude) ne kadar değiştiğini gösteren skaler değerdir. 
# Yani, özdeğer, özvektörün matrisle etkileşim sonucunda ne kadar büyüyüp küçüldüğünü ifade eder.

# Formül     _ 
#        y - y = y   kendilerine eşitliyorum böylece  ortalamalarını çıkarttığım için veriyi 0 merkezli hale getiriyorum artık verinin merkezi  0.0 noktası

# Kovaryans :     iki değişkenin birbirine olan doğrusal ilişkisini ölçen bir değerdir. ( NE KADAR DEĞİŞTİĞİNİ ÖLÇER )



#minik örnek

xx = [2.4,0.6,2.1,2,3,2.5,1.9,1.1,1.5,1.2]
yy = [2.5,0.7,2.9,2.2,3.0,2.3,2.0,1.1,1.6,0.8]

xx = np.array(xx)
yy = np.array(yy)
plt.scatter(xx,yy)   # gördüğümüz gibi 0 merkezli değil ilk adım 0 merkezli yapmaktı

xx_m = np.mean(xx)
yy_m = np.mean(yy)

xx = xx-xx_m
yy = yy-yy_m
plt.scatter(xx,yy)   # 0 merkezli hale geldi , şimdi kovaryansını bulmam gerekiyor

cov = np.cov(xx,yy)
print(cov)
# Cov= [ Cov(x,x) Cov(y,x) ]
#      [ ​Cov(x,y) Cov(y,y) ]

# kovaryans değerlerini bulduk şimdi eigenvalue ve eigenvector değerlerini bulmak için alttaki kütüphaneyi çağırmamız gerek

from numpy import linalg as LA
w,v= LA.eig(cov)
print("W : ",w)   # eigenvalue
print()
print("V : ",v)   # eigenvector

# burda bunları nasıl barındırıyor ?  bu sayılar neye denk geliyor ?

#  Eigenvalue'lar(W), verinin varyansının en yüksek olduğu yönleri gösterir.
# 0.04215805: Bu, ilk eigenvalue'dur. Bu küçük değer, verinin birinci yönündeki varyansın çok düşük olduğunu gösterir.
# 1.18117528: Bu, ikinci eigenvalue'dur. Bu yüksek değer, verinin ikinci yönündeki varyansının daha büyük olduğunu, yani bu yönde daha fazla bilgi olduğunu gösterir.

# Özetle: Bu iki eigenvalue, x ve y veri setlerinin dönüşümden sonra verinin hangi yönlerde ne kadar yayıldığını (varyansı) ifade eder.
# İkinci yön (1.181) verideki varyansın çoğunu taşırken, birinci yön (0.042) çok az varyansa sahiptir.

# Eigenvector'ler (V):
# Bu matris, her bir eigenvalue'ya karşılık gelen eigenvector'leri içerir. Her bir sütun bir eigenvector'dür.

# İlk eigenvector:    [-0.75410555, 0.65675324]
# Bu eigenvector, ilk eigenvalue'ya karşılık gelir (yani 0.04215805). Verinin az yayıldığı yönü (daha düşük varyans) gösterir.


# İkinci eigenvector: [-0.65675324, -0.75410555]
# Bu eigenvector, ikinci eigenvalue'ya karşılık gelir (yani 1.18117528). Bu vektör, verinin en çok yayıldığı (en yüksek varyanslı) yönü gösterir.
# PCA'da bu genellikle ana bileşen olarak seçilir.

# Özetle: Eigenvector'ler, veriyi hangi yönlere döndürmemiz gerektiğini gösterir.
# İlk eigenvector, daha az varyansa sahip bir yönü, ikinci eigenvector ise daha çok varyansa sahip bir yönü temsil eder.

# Varyans, bir veri kümesindeki değerlerin ortalama etrafında ne kadar dağıldığını ya da yayıldığını ölçer.
# Varyans ne kadar fazlaysa veriler arasındaki boşluk o kadar fazladır  
# Varyans ile veri miktarı arasında doğrudan bir ilişki yoktur.

# Diyelim ki iki farklı veri setiniz var:
# Veri Seti A: [1, 2, 1, 2, 1] (Düşük varyans)       Veri Seti A: Bu setin varyansı düşüktür çünkü veriler birbirine çok yakın.
# Veri Seti B: [1, 5, 3, 8, 2] (Yüksek varyans)      Veri Seti B: Bu setin varyansı yüksektir çünkü veriler arasında büyük farklılıklar vardır.


p1 = v[:,1]
p2 = v[:,0]

plt.plot([-3*p1[0],3*p1[0]],[-3*p1[1],3*p1[1]])  # normalde böyle denedik ilk başta plt.plot([0,p1[0]],[0,p1[1]])  ama boyutunu büyütmek için *ile çarptık

plt.plot([-3*p2[0],3*p2[0]],[-3*p2[1],3*p2[1]]) 


#%%  PCA İLE standartize etme

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)   # pca algoritması unspervised learning algoritması class label ihtiyaç duymuyor bu nedenle sadece x_train 
                                        # kullanmaktansa tüm x inputlarını scale edicez yani ana verimizi train test diye ayırmıycaz

pca = PCA(n_components=2)     #diziyi 2 boyutlu hale çeviriyor
pca.fit(x_scaled)
pca.transform(x_scaled)


X_reduced_pca = pca.transform(x_scaled)
pca_data = pd.DataFrame(X_reduced_pca,columns=["p1","p2"])
pca_data["target"] = y
sns.scatterplot(x = "p1",y="p2",hue="target",data=pca_data)
plt.title("PCA : P1 VS P2")

# 30 BOYUTLU VERİMİZİ 2 BOYUTLU HALE GETİRDİK VE GÖRSELLEŞTİRDİK
# ŞİMDİ 2 BOYUTLU ALGORİTMAMLA KNN SINIFLANDIRMA İŞLENİNİ GERÇEKLEŞTİRİCEM

X_train_pca,X_test_pca,Y_train_pca,Y_test_pca = train_test_split(X_reduced_pca,y,test_size=0.3,random_state=42)

grid_pca = KNN_best_parameters(X_train_pca, X_test_pca, Y_train_pca, Y_test_pca)


# BURAYI HAZIR ALDIK AMACIMIZ GÖRSELLEŞTİRMEK

cmap_light = ListedColormap(["orange","cornflowerblue"])
cmap_bold = ListedColormap(["darkorange","darkblue"])

h = .05
X = X_reduced_pca
x_min,x_max = X[:,0].min() -1 ,X[:,0].max() +1
y_min,y_max = X[:,1].min() -1 ,X[:,1].max() +1

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

#%%   NCA

#  veri boyutunu azaltmak ve sınıflandırma problemlerini daha etkili hale getirmek için kullanılan bir yöntemdir
# Denetimli Öğrenme: NCA, verilerin sınıf etiketlerini kullanarak öğrenir. 
# Bu nedenle, sınıflar arasındaki ayrımı optimize etmek amacıyla verilerin komşuluk ilişkilerini dikkate alır.


nca = NeighborhoodComponentsAnalysis(n_components=2,random_state=42)
nca.fit(x_scaled,y)
X_reduced_nca = nca.transform(x_scaled)
nca_data = pd.DataFrame(X_reduced_nca,columns=["p1","p2"])
nca_data["target"] = y
sns.scatterplot(x="p1",y="p2",hue="target",data=nca_data)
plt.title("NCA : P1 VS P2")


X_train_nca,X_test_nca,Y_train_nca,Y_test_nca = train_test_split(X_reduced_nca,y,test_size=0.3,random_state=42)

grid_nca = KNN_best_parameters(X_train_nca, X_test_nca, Y_train_nca, Y_test_nca)

# hatayı bulmak için görselleştirelim

cmap_light = ListedColormap(["orange","cornflowerblue"])
cmap_bold = ListedColormap(["darkorange","darkblue"])

h = .2
X = X_reduced_nca
x_min,x_max = X[:,0].min() -1 ,X[:,0].max() +1
y_min,y_max = X[:,1].min() -1 ,X[:,1].max() +1

xx ,yy = np.meshgrid(np.arange(x_min,x_max,h),
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


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    