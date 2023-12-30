#Yasemin Yavuz Kaya 20231530001
#İstatiksel Yapay Öğrenme
#İŞ PROBLEMİ B


###Cevap 1

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, f_oneway, mannwhitneyu, kruskal, pearsonr, chi2_contingency, spearmanr


###Cevap 2
df = pd.read_csv('email_marketing.csv')

###Cevap 3

#3a

df.head(10)

#   Musteri_ID    Kampanya  Eposta_Acilma_Orani  Tıklama_Orani  Satin_Alma_Miktari Musteri_Segmenti    Bolge
#0           1  Kampanya A                90.83          77.71                   3    Düşük Harcama  Bölge 4
#1           3  Kampanya A                14.49          55.84                   4     Yeni Müşteri  Bölge 3
#2           4  Kampanya A                48.95          42.42                  12      Sık Müşteri  Bölge 1
#3           5  Kampanya A                98.57          90.64                  18    Düşük Harcama  Bölge 2
#4           7  Kampanya A                67.21          11.12                  19    Düşük Harcama  Bölge 3
#5           8  Kampanya A                76.16          49.26                  17      Sık Müşteri  Bölge 1
#6           9  Kampanya A                23.76           1.14                   1     Yeni Müşteri  Bölge 2
#7          11  Kampanya A                36.78          46.87                  27     Yeni Müşteri  Bölge 4
#8          12  Kampanya A                63.23           5.63                   8     Yeni Müşteri  Bölge 4
#9          13  Kampanya A                63.35          11.88                  23      Sık Müşteri  Bölge 1

df.tail(10)

#     Musteri_ID    Kampanya  Eposta_Acilma_Orani  Tıklama_Orani  Satin_Alma_Miktari Musteri_Segmenti    Bolge
#190         171  Kampanya B                54.92          42.54                  12    Düşük Harcama  Bölge 3
#191         174  Kampanya B                27.99          50.76                   2    Düşük Harcama  Bölge 2
#192         175  Kampanya B                95.49          24.24                  15     Yeni Müşteri  Bölge 3
#193         176  Kampanya B                73.79          11.48                   6      Sık Müşteri  Bölge 3
#194         183  Kampanya B                 1.44          61.06                   7     Yeni Müşteri  Bölge 3
#195         187  Kampanya B                85.55          28.86                  26      Sık Müşteri  Bölge 1
#196         190  Kampanya B                 9.78          58.12                  16     Yeni Müşteri  Bölge 1
#197         196  Kampanya B                61.59          15.44                  13      Sık Müşteri  Bölge 4
#198         197  Kampanya B                63.51          48.11                  10    Düşük Harcama  Bölge 2
#199         198  Kampanya B                 4.53          53.26                   4      Sık Müşteri  Bölge 4

##3b

df['Kampanya'].unique()
df.groupby('Kampanya').agg({'Satin_Alma_Miktari': 'mean'})

#            Satin_Alma_Miktari
#Kampanya
#Kampanya A               15.02
#Kampanya B               16.01

#Kampanya B'de satın alma miktarı ortalama oranı daha fazladır.

###Cevap 4

#Parametrik testler için varsayımlar


# Normallik Varsayımı
# Varyans Homojenliği

#Normallik Varsayımı (shapiro)


# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.

# Eğer p-değeri 0.01'ten küçükse, null hipotezi (H0) reddedilir.
# Eğer p-değeri 0.01'ten büyük veya eşitse, null hipotezi (H0) reddedilemez.


shapiro(df.loc[df['Kampanya']== 'Kampanya A', 'Satin_Alma_Miktari'])
#Out[4]: ShapiroResult(statistic=0.9765703678131104, pvalue=0.07163432985544205)

shapiro(df.loc[df['Kampanya']== 'Kampanya B', 'Satin_Alma_Miktari'])
#Out[5]: ShapiroResult(statistic=0.9722954034805298, pvalue=0.03321434557437897)

#### p-değeri 0.01'ten büyük, null hipotezi (H0) reddedilemez.
# H0: Normal dağılım varsayımı sağlanmaktadır.


#Varyans Homojenliği Varsayımı(levene)

# H0: Grupların varyansları homojendir.
# H1: Grupların varyansları homojen değildir.

levene(df.loc[df['Kampanya']== 'Kampanya A', 'Satin_Alma_Miktari'],
       df.loc[df['Kampanya']== 'Kampanya B', 'Satin_Alma_Miktari'])
#Out[6]: LeveneResult(statistic=0.2637482361616668, pvalue=0.608129283537828)
#### p-değeri 0.01'ten büyük, null hipotezi (H0) reddedilemez.
# H0: Grupların varyansları homojendir.

#4b

# Nihai Test
# Normallik varsayimi ve Varyans homojenligi varsayimi birlikte saglandigi durumda: (2 grup karsilastirmasi icin t testi, 2+ grup karsilastirmasi icin ANOVA)


ttest_ind(df.loc[df['Kampanya']== 'Kampanya A', 'Satin_Alma_Miktari'],
       df.loc[df['Kampanya']== 'Kampanya B', 'Satin_Alma_Miktari'],
          equal_var=True)
#Out[8]: TtestResult(statistic=-0.9273452677572074, pvalue=0.35487631887778204, df=198.0)

#### p-değeri 0.01'ten büyük, null hipotezi (H0) reddedilemez.

# H0: Kampanya A ve Kampanya B'de  ortalama toplam satınalma miktarı arasında istatistiksel olarak anlamlı bir fark yoktur.

#Cevap 5

#Parametrik testler için varsayımlar


# Normallik Varsayımı
# Varyans Homojenliği

#Normallik Varsayımı (shapiro)


# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.

# Eğer p-değeri 0.05'ten küçükse, null hipotezi (H0) reddedilir.
# Eğer p-değeri 0.05'ten büyük veya eşitse, null hipotezi (H0) reddedilemez.


shapiro(df.loc[df['Kampanya']== 'Kampanya A', 'Satin_Alma_Miktari'])
#Out[4]: ShapiroResult(statistic=0.9765703678131104, pvalue=0.07163432985544205)

shapiro(df.loc[df['Kampanya']== 'Kampanya B', 'Satin_Alma_Miktari'])
#Out[5]: ShapiroResult(statistic=0.9722954034805298, pvalue=0.03321434557437897)

# Normallik varsayimi saglanmiyor, direkt Non-parametrik teste gecilmelidir.

mannwhitneyu(df.loc[df['Kampanya']== 'Kampanya A', 'Satin_Alma_Miktari'],
       df.loc[df['Kampanya']== 'Kampanya B', 'Satin_Alma_Miktari'])

#Out[9]: MannwhitneyuResult(statistic=4649.0, pvalue=0.39118576064396826)
# p-değeri 0.05'ten büyük, null hipotezi (H0) reddedilemez.
# H0: Kampanya A ve Kampanya B'de  ortalama toplam satınalma miktarı arasında istatistiksel olarak anlamlı bir fark yoktur.

#Ödevi seçme nedenim kampanyalar arasındaki satınalma miktarlarını merak etmemdir.