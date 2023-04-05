#!/usr/bin/env python
# coding: utf-8

# # Проект подготовили Лабуть Эвелина и Хорасанджян Левон

# # Исследовательский проект по банковскому маркетинговому датасету

# ---

# ## Переменные

# **age -** возраст клиента - количественная дискретная переменная - integer
# 
#  **job** ('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')  - тип работы клиента - категориальная номинальная переменная - string
# 
# **marital** ('divorced','married','single','unknown'; note: 'divorced' means divorced or widowed) - семейное положение клиента - категориальная номинальная переменная - string
# 
# **education** ('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown') - уровень образования клиента - категориальная номинальная переменная - string
# 
# **default** ('no','yes','unknown') - есть ли неуплата по кредиту - категориальная бинарная переменная - string
# 
# **housing** ('no','yes','unknown') - есть ли жилищный кредит  - категориальная бинарная переменная - string
# 
# **loan** ('no','yes','unknown')- есть ли потребительский кредит  - категориальная бинарная переменная - string
# 
# **contact** ('cellular','telephone') - вид взаимодействия с клиентом - категориальная номинальная переменная - string
# 
# **month** (categorical: 'jan', 'feb', 'mar', …, 'nov', 'dec') - месяц последнего разговора - категориальная номинальная переменная - string
# 
# **day*of*week** ('mon','tue','wed','thu','fri') - день недели последнего разговора - категориальная номинальная переменная - string
# 
# **duration** - длительность последнего разговора в секундах - количественная дискретная переменная - integer
# 
# **campaign** - количество разговоров с клиентом во время текущей маркетинговой компании - количественная дискретная переменная - integer
# 
# **pdays** (999 means client was not previously contacted) - количество дней, прошедших с момента разговора в рамках предыдущей маркетинговой компании - количественная дискретная переменная - integer
# 
# **previous** - количество звонков клиенту в рамках предыдущей маркетинговой компании - количественная дискретная переменная - integer
# 
# **poutcome** ('failure','nonexistent','success') -  результат предыдущей маркетинговой компании - категориальная номинальная переменная - string
# 
# **emp.var.rate** - коэффициент изменения трудоустройства в рамках текущего квартала - количественная непрерывная переменная - decimal
# 
# **cons.price.idx** - индекс потребительских цен в рамках текущего месяца - количественная непрерывная переменная - decimal
# 
# **cons.conf.idx** - индекс потребительского доверия в рамках текущего месяца - количественная непрерывная переменная - decimal
# 
# **euribor3m** - Европейская межбанковская ставка предложения на данный момент - количественная непрерывная переменная - decimal
# 
# **nr.employed** - количество сотрудников в рамках текущего квартала - количественная  дискретная переменная - decimal
# 
# **subscribed** ('yes','no') - оформил ли клиент срочный вклад - категориальная бинарная переменная - boolean

# ---

# ## Работа с таблицей и её обработка

# In[1]:


# загружаем датасет и подключаем библиотеки
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('bank_marketing_dataset.csv')
data


# In[2]:


# удалим не нужные для анализа колонки, а так же все строки с параметром "unknown" в одном из столбцов
data.drop(['emp.var.rate',  'nr.employed','default'], axis=1, inplace=True)
cols = ['job','education','housing','loan','marital']
for c in cols:
  data = data.drop(np.where(data[c] == 'unknown')[0])
  data.reset_index(drop=True, inplace=True)
data


# In[3]:


# для удобства поменяем тип некоторых категориальных и бинарных переменных на целочисленный
binary_cols = ['housing', 'loan', 'subscribed']
value_list = []
for c in binary_cols:
    for text in data[c].values:
        if text == 'yes':
            value_list.append(1)
        else:
            value_list.append(0)
    data[c] = value_list
    value_list = []
data


# In[4]:


# создадим новую колонку has_any_loan - есть ли у клиента какой-либо кредит 
data["sum"] = data["loan"]+data["housing"]
data.insert(6,"has_any_loan" , data["sum"].apply(lambda x: 1 if x>=1 else 0))
data.drop(["sum"],axis=1,inplace = True)
data 


# In[5]:


# разобьём переменную age на 6 категорий ("17-24", "25-29", "30-44", "45-54", "55-64", "65+")
# результаты запишем в новую колонку age_group
# этот показатель пригодится для проверки гипотез, основанных на возрастном промежутке клиентов
def age_func(x):
  if x<25:
    return "17-24"
  elif x<30:
    return "25-29"
  elif x<45:
    return "30-44"
  elif x<55:
    return "45-54"
  elif x<65:
    return "55-64"
  else:
    return "65+"
data.insert(0,'age_group',data['age'].apply(age_func))
data


# In[6]:


# сформируем транспонированный срез по таблице (первые 20 клиентов и 8 колонок)
sliced_data = data.iloc [0:20, 0:8]
sliced_data = sliced_data.T
print(sliced_data)


# In[7]:


# отсортируем значения по двум парметрам : возраст и индекс потребительских цен 
sorted_df = data.sort_values(by=['age','cons.price.idx'], ascending=True)
print(sorted_df.head(20))


# ---

# ## Гипотезы

# ### Гипотеза 1
# 
# Люди с высшим образованием оформляют потребительский кредит реже остальных

# In[8]:


# библиотека для построения графиков
#%pip install plotly 
import plotly.graph_objs as go
# массив с цветами
colors = ["#BDCBEA","#7A8DF4","#FCC03E","#F7B487","#CBC8D2","#DEDCE1","#FFDD93","#A3BDFF","#F1DBCE"]


# In[9]:


print("Гипотеза 1")
fig1 = go.Figure(layout = {'paper_bgcolor':"#2B4B82",'font':{'color':'white','size':16},'colorway':colors})
fig1.add_trace(go.Pie(values=data.loan, labels=data.education,textfont=dict(size=16,color='white')))
fig1.show()


# Итог: гипотеза полностью опровергнута - люди с высшим образованием оформляют потребительский кредит ЧАЩЕ остальных

# ### Гипотеза 2
# 
# С пожилыми людьми (65+ лет) чаще связывались по стационарному телефону

# In[10]:


print("Гипотеза 2")
graphic_2 = data[data.age_group=='65+'].groupby(data.contact).size().plot(kind="bar", width=0.3, figsize=(5, 5), color="#7B8CF3")
graphic_2.set_xlabel("Contact's type")
graphic_2.set_ylabel("Number of people")


# Итог: гипотеза вновь опровергнута - с большим перевесом люди в возрасте 65+ пользуются сотовым примерно в 6 раз чаще, чем стационарным телефоном

# ### Гипотеза 3
# 
# Люди, состоящие в браке, оформляют жилищный кредит чаще одиноких

# In[11]:


fig3 = go.Figure(layout = {'paper_bgcolor':"#2B4B82",'font':{'color':'white','size':16},'colorway':colors})
fig3.add_trace(go.Pie(values=data.housing, labels=data.marital,textfont=dict(size=16,color='white')))
fig3.show()
print("Гипотеза 3")


# Итог: гипотеза подтвердилась - люди в браке действительно оформляют жилищный кредит чаще, чем одинокие

# ### Гипотеза 4
# 
# Чаще всего кредиты оформляют люди, деятельность которых связана с администрированием

# In[12]:


graphic_4 = data.groupby("job").count()['has_any_loan'].plot(kind="bar", width=0.8, figsize=(5, 5), color="#7B8CF3")
graphic_4.set_xlabel("Job")
graphic_4.set_ylabel("Number of people")
print("Гипотеза 4")


# Итог: гипотеза подтвердилась - люди, деятельность которых связана с администрированием, действительно оформляют кредиты чаще остальных

# ### Гипотеза 5
# 
# Звонки клиентам чаще всего совершались по понедельникам

# In[13]:


print("Гипотеза 5")
fig5 = go.Figure(layout = {'paper_bgcolor':"#2B4B82",'font':{'color':'white','size':16},'colorway':colors})
fig5.add_trace(go.Pie( labels = data.day_of_week,textfont=dict(size=16,color='white')))
fig5.show()


# Гипотеза оказалась не верной. Как оказалось, большинство звонков совершалось в четверг, причем число звонков в каждый из дней недели практически одинаково.

# ### Гипотеза 6
# 
# Возрастной промежуток наиболее частого оформления кредитов 30-44 лет

# In[14]:


graphic_6 = data[data.has_any_loan == 1].age_group.hist(bins=75, color="#7B8CF3")
graphic_6.set_xlabel("Age")
graphic_6.set_ylabel("Number of people")
print("Гипотеза 6")


# Итог: гипотеза подтвердилась - чаще всего кредит оформляется в возрасте от 30 до 44 лет

# ---

# ## Сводные таблицы

# ### Первая сводная таблица

# In[15]:


table_1 = pd.pivot_table(data, index=[data.age_group], values=["duration"])
table_1


# Исходя из показателей сводной таблицы по возрастной группе длительности звонков, можно сделать вывод, что наиболее длительные разговоры ведутся с клиентами из возрастной группы 65+

# ### Вторая сводная таблица

# In[16]:


table_2 = pd.pivot_table(data, index=['month'], values=["subscribed"])
table_2


# Из второй сводной таблицы приходим к выводу, что март стал наиболее продуктивным месяцем рекламной компании (с большинством оформивших кредит связывались в марте). Также отметим не сильно отстающие месяцы: сентябрь, октябрь и декабрь

# ### Третья сводная таблица

# In[17]:


table_3 = pd.pivot_table(data, index=[data.poutcome], values=["cons.price.idx", "euribor3m"])
table_3.drop("nonexistent")


# Из третьей сводной таблицы можно заметить, что самый низкий индекс доверия наблюдался во времена провальных кампаний. Помимо этого, европейская межбанковская ставка в провальные кампании была выше, чем в успешные, примерно на 65%

# ---

# ## Описательные статистики

# In[18]:


describe_data = data.describe()
describe_data.drop(["housing","loan","has_any_loan","subscribed"], axis=1, inplace=True)
describe_data


# Выводы на основе из таблицы выше:
# - Минимальный возраст клиентов 17 лет, максимальный - 98 лет, средний - 39-40 лет.
# - Средняя длительность последнего разговора с клиентом - 258 секунд (приблизительно 4 минуты).
# - Индекс потребительской уверенности упал в среднем на 40% за текущий месяц.
# - Индекс потребительских цен вырос в среднем на 93%.
# - Во время текущей маркетинговой компании с клиентом в среднем производилось 2 разговора, хотя максимальное количество звонков клиенту - 43.

# In[19]:


import statistics as stat
print(stat.mode(data["month"]))


# Вывод:  месяцем наиболее активной рекламной кампании был май

# In[20]:


print(stat.mean(data[data.subscribed==1]["campaign"]))


# Вывод: клиентам, оформившим кредит, на протяжении текущей рекламной компании звонили в среднем два раза

# ---

# ## Корреляция

# In[21]:


# посчитаем корреляцию между данными 
correlated_data = data.corr()
plt.figure(figsize= (12,8))
sns.heatmap(correlated_data, cmap="Blues", annot=True)
print("корреляция")


# Выводы по корреляции: 
# - Самая высокая корреляция (0.88) наблюдается между показателями "has_any_loan" и "housing", из чего можно предположить, что по всем видам кредита жилищный является самым распространённым. 
# - Чуть менее сильная корреляция (0.69) наблюдается и между показателями "euribor3m" и "cons.price.idx", что означает большую зависимость индекса потребительских цен от европейской межбанковской ставки предложения. 
# - Из отрицательных корреляций самая высокая (-0.58) и очевидная наблюдается между "previous" и "pdays", то есть чем больше прошло дней с момента последнего разговора с клиентом, тем реже совершались звонки в рамках предыдущей маркетинговой кампании. 
# - Вторая самая высокая отрицательная корреляция между "euribor3m" и "previous", то есть чем выше оказывалась европейской межбанковской ставки предложения, тем реже связывались с клиентами в рамках предыдущей маркетинговой кампании.

# ---

# ## Общие выводы по результатам анализа данных

# По нашим результатам из 6 выдвинутых гипотез подтвердилось следующие три: 
# - люди в браке оформляют жилищный кредит чаще одиноких;
# - люди, деятельность которых связана с администрированием, оформляют кредиты чаще других; 
# - 30-44 лет — это частый возрастной промежуток кредитного займа.
# 
# Из опровергнутых гипотез стало известно, что:
# - люди с высшим образованием оформляют потребительский кредит чаще остальных;
# - с пожилыми гражданами в возрасте от 65 лет связывались по сотовому телефону примерно в 6 раз чаще, чем по стационарному;
# - большинство звонков клиентам совершалось в четверг.

# ------

# # Web Scraping сайта Nike.com

# In[22]:


#pip install bs4


# In[23]:


#подключаем библиотеки
from bs4 import BeautifulSoup
import requests
import xlrd
import pandas as pd


# In[24]:


categories = ['Спортивный стиль','Бег','Баскетбол','Фитнес','Футбол','Скейтбординг']
# страницы, с которых будем собирать ссылки на другие страницы 
links = ['https://www.nike.com/ru/w/mens-lifestyle-shoes-13jrmznik1zy7ok','https://www.nike.com/ru/w/mens-running-shoes-37v7jznik1zy7ok','https://www.nike.com/ru/w/mens-basketball-shoes-3glsmznik1zy7ok','https://www.nike.com/ru/w/mens-training-gym-shoes-58jtoznik1zy7ok','https://www.nike.com/ru/w/mens-soccer-shoes-1gdj0znik1zy7ok','https://www.nike.com/ru/w/mens-skateboarding-shoes-8mfrfznik1zy7ok']


# In[25]:


# датафрейм для хранения данных
df = pd.DataFrame()


# In[26]:


# парсим названия товаров с общих страниц
names = []
category = []
for i in range(len(links)):
  answer_from_nike = requests.get(links[i])
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  div_titles = soup_page.find_all('div',{'class':'product-card__title'})
  for title in div_titles:
    names.append(title['id'])
    category.append(categories[i])
df["Название модели"] = names
df["Категория"] = category
df


# In[27]:


# парсим цены на товары с общих страниц
prices = []
def conv_to_int(str_element):
  str_element = list(str_element)
  i = 0
  while i<len(str_element):
    if not str_element[i].isdigit():
      str_element.remove(str_element[i])
      i -= 1
    i+=1
  return str_element
for i in range(len(links)):
  answer_from_nike = requests.get(links[i])
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  product_info= soup_page.find_all('div',{'class':'product-card__info'})
  for line in product_info:
    price = line.find('div',{'class':'product-price'}).text
    price = conv_to_int(price)
    prices.append(int(''.join(price)))
df["Цена"] = prices
df


# In[28]:


# парсим количество доступных цветов с общих страниц
colors = []
for i in range(len(links)):
  answer_from_nike = requests.get(links[i])
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  product_info= soup_page.find_all('div',{'class':'product-card__info'})
  for line in product_info:
    colors_count = line.find('div',{'class':'product-card__product-count'}).text
    colors_count = conv_to_int(colors_count)
    colors.append(int(''.join(colors_count)))
df["Количество цветов"] = colors
df


# In[29]:


# парсим ссылки на каждую модель
product_links = []
for l in links:
  answer_from_nike = requests.get(l)
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  div_titles = soup_page.find_all('a',{'class':'product-card__link-overlay'})
  for title in div_titles:
    product_links.append(title['href'])
df['Ссылка на модель'] = product_links
df


# In[30]:


#парсим номер отдельно для каждой модели по ссылке из списка
model_tag = []
for l in product_links:
  answer_from_nike = requests.get(l)
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  li_titles = soup_page.find('li',{'class':"description-preview__style-color ncss-li"})
  tag = str(li_titles)[61:71]
  model_tag.append(tag)
df["Номер модели"] = model_tag
df


# In[31]:


#парсим список цветов отдельно для каждой модели по ссылке из списка 
from re import L
def parse_colors(tag):
  tag = str(tag)[72::]
  index = tag.find('<')
  tag = tag[:index]
  l = list(set(tag.split('/')))
  tag = ",".join(l)
  return tag
colors_list = []
for l in product_links:
  answer_from_nike = requests.get(l)
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  li_titles = soup_page.find('li',{'class':"description-preview__color-description ncss-li"})
  colors_list.append(parse_colors(li_titles))
df.insert(4,'Цвета модели',colors_list)
df


# In[32]:


# парсим количество отзывов отдельно на каждую модель по ссылке из списка 
review = []
def parse_review(tag):
  tag = str(tag)[29::]
  index = tag.find(')')
  tag = tag[:index]
  if(len(tag)==0):
    return 0
  else:
    return int(''.join(conv_to_int(tag)))
for l in product_links:
  answer_from_nike = requests.get(l)
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  h3_titles = soup_page.find('h3',{'class':"css-xd87ek"})
  review.append(parse_review(h3_titles))
df["Количество отзывов"] = review
df


# In[33]:


# парсим особенности отдельно для каждой модели по ссылке из списка 
features = []
def parse_features(line):
  line = str(line)[50::]
  index = line.find('<')
  return line[:index]
for l in product_links:
  answer_from_nike = requests.get(l)
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  title = soup_page.find('div',{'class':'headline-5 text-color-accent d-sm-ib'})
  if(title!=None):
    features.append(parse_features(title))
  else:
    features.append("Отсутствуют")
df["Особенности"] = features
df


# In[34]:


# парсим рейтинг отдельно для каждой модели по ссылке из списка
rating_num = []
def parse_rating(tag):
  tag = str(tag)[26::]
  index = tag.find('<')
  tag = tag[:index]
  return tag
for l in product_links:
  answer_from_nike = requests.get(l)
  soup_page = BeautifulSoup (answer_from_nike.text,"html.parser")
  p_titles = soup_page.find('p',{'class':"d-sm-ib pl4-sm"})
  if p_titles!=None:
    rating_num .append(parse_rating(p_titles))
  else:
    rating_num.append("0")
df["Рейтинг"] = rating_num
df = df.astype({'Рейтинг':'float'})
df


# In[35]:


writer = pd.ExcelWriter('nike_web_scraping.xlsx')
df.to_excel(writer)
writer.save()

