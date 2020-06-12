import pandas as pd
import calendar
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import folium
from folium import plugins
from folium.plugins import HeatMapWithTime
import numpy as np

df_2019 = pd.read_csv('2019_accidents.csv')
print(df_2019.head())

print(list(df_2019.columns))

print(df_2019.info())

# Виявлення нульових значень
print(df_2019.isnull().sum().any())
df_2019.replace('Unknown', np.nan, inplace=True)
print(df_2019.isnull().sum().any())
print(df_2019.info())

# Типи даних
print(df_2019.dtypes)

# Зміна назв ( пробіл на _ )
df_2019.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
print(df_2019.columns)

# Робимо з окремих стовпчиків один з датою (date)
df_2019.rename(columns={'nk_any': 'year', 'mes_any': 'month', 'dia_mes': 'day', 'hora_dia': 'hour'}, inplace=True)
df_2019['date'] = pd.to_datetime(df_2019[['year', 'month', 'day', 'hour']])
print(df_2019.head())
print(df_2019.date.dtypes)

df_2019.rename(columns={'numero_expedient': 'id'}, inplace=True)

# Від прогавин в кінці id
print(df_2019.id.loc[0])
df_2019.id = df_2019.id.apply(lambda x: x.strip())
print(df_2019.id.loc[0])

# Встановлення id в якості індексу в датасеті
df_2019.set_index('id', inplace=True)
print(df_2019.loc['2019S000001'])

# Пошук дублікатів в датасеті
print(df_2019.duplicated().sum())
print(df_2019[df_2019.duplicated()])

# Видалення дублікатів
print(df_2019.shape)
df_2019.drop_duplicates(inplace=True)
print(df_2019.shape)

print('Total number of accidents in 2017 :{}'.format(df_2019.shape[0]))

# Деякі підсумки
mild_inj = df_2019["numero_lesionats_lleus"].sum()
seri_inj = df_2019["numero_lesionats_greus"].sum()
vehi = df_2019["numero_vehicles_implicats"].sum()
victims = df_2019['numero_victimes'].sum()

print('Total Number of vehicles involved:', vehi)
print('Total Number of Victims:', victims)
print('Total Number of Victims with mild injuries:', mild_inj)
print('Total Number of Victims with serious injuries:', seri_inj)

# Кількість аварій в місяць
accidents_month = df_2019.groupby(df_2019['date'].dt.month).count().date
accidents_month.index = [calendar.month_name[x] for x in range(1, 13)]
print(accidents_month)

t_accident = accidents_month.tolist()
t_month = list(accidents_month.index)

fig = go.Figure(data=[go.Bar(x=["Січень", "Лютий", "Березень", "Квітень", "Травень", "Червень", "Липень", "Серпень", "Вереснеь", "Жовтень", "Листопад", "Грудень"], y=t_accident,
                             marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                             marker_line_width=1.5, opacity=0.6,
                             text=t_accident, textposition='outside')])
fig.update_layout(title='Аварії в Барселоні в 2019 році',
                  yaxis_title='Кількість аварій в місяць', xaxis_title='Місяць')
fig.show()

# Кількість аварій в день тижня
accidents_day = df_2019.groupby(df_2019['date'].dt.dayofweek).count().date
accidents_day.index = [calendar.day_name[x] for x in range(0, 7)]

t_accident_week = accidents_day.tolist()
t_day = list(accidents_day.index)

fig1 = go.Figure(data=[go.Bar(x=["Понеділок", "Вівторок", "Середа", "Четвер", "П'ятниця", "Субота", "Неділя"], y=t_accident_week,
                              text=t_accident_week, textposition='outside',
                              marker_color='rgb(255,182,193)', marker_line_color='rgb(219,112,147)',
                              marker_line_width=1.5, opacity=0.6)])
fig1.update_layout(title='Аварії в Барселоні в 2019 році',
                   yaxis_title='Кількість аварій в день тижня', xaxis_title='День тижня')
fig1.show()

# Количество аварій в годину
accidents_hour = df_2019.groupby(df_2019['date'].dt.hour).count().date
fig4 = go.Figure(data=[go.Bar(x=accidents_hour.index, y=accidents_hour, marker_color=['red'] * 24, width=0.5,
                              marker_line_color='rgb(0,0,0)', text=accidents_hour, textposition='outside',
                              opacity=0.6)])
fig4.update_layout(title='Аварії в Барселоні в 2019 році',
                   yaxis_title='Кількість аварій в конкретну годину', xaxis_title='Година')
fig4.show()

# Співвідношення тяжкості травм
injuries = df_2019[['numero_lesionats_lleus', 'numero_lesionats_greus']].sum()
fig5 = go.Figure(data=[
    go.Pie(values=injuries, title='Аварії в Барселоні в 2019 році', textinfo='label', opacity=0.6, titleposition='top left',
           titlefont=dict(
               size= 15,
               color='rgb(0,0,128)'
           ),
           labels=['Легкі травми', 'Важкі травми'], hole=.3, pull=[0, 0.18])])
fig5.update_traces(hoverinfo='label+percent', textinfo='value+percent', textfont_size=20,
                   marker=dict(line=dict(color='rgb(0,0,128)', width=2)))
fig5.show()

# Кількість серйозних травм в конкретний день тижня
accidents_serious = df_2019[df_2019['numero_lesionats_greus'] != 0].groupby(
    df_2019['date'].dt.dayofweek).sum().numero_lesionats_greus
rate_serious = accidents_serious / accidents_serious.sum()

accidents_mild = df_2019[df_2019['numero_lesionats_lleus'] != 0].groupby(
    df_2019['date'].dt.dayofweek).sum().numero_lesionats_lleus
rate_mild = accidents_mild / accidents_mild.sum()

rates = pd.DataFrame({'Тяжкі травми': rate_serious, 'Легкі травми': rate_mild})
rates.plot(kind='bar', figsize=(12, 7), color=['red', 'green'], alpha=0.5)

plt.title('Кількість травм на дорогах в Барселоні в 2019', fontsize=20)
plt.xlabel('День тижня', fontsize=16)
plt.ylabel('Відсоток', fontsize=16)
plt.xticks(np.arange(7), ["Понеділок", "Вівторок", "Середа", "Четвер", "П'ятниця", "Субота", "Неділя"]);
plt.show()

# Кількість серйозних травм о конкретній годині
accidents_serious = df_2019[df_2019['numero_lesionats_greus'] != 0].groupby(
    df_2019['date'].dt.hour).sum().numero_lesionats_greus
rate_serious = accidents_serious / accidents_serious.sum()
accidents_mild = df_2019[df_2019['numero_lesionats_lleus'] != 0].groupby(
    df_2019['date'].dt.hour).sum().numero_lesionats_lleus
rate_mild = accidents_mild / accidents_mild.sum()
rates = pd.DataFrame({'Тяжкі травми': rate_serious, 'Легкі травми': rate_mild})
rates.plot(kind='bar', figsize=(12, 7), color=['red', 'green'], alpha=0.5)
plt.title('Кількість травм на дорогах в Барселоні в 2019', fontsize=20)
plt.xlabel('Година', fontsize=16)
plt.ylabel('Відсоток', fontsize=16)
plt.show()

# Нещасні випадки на карті з серйозними травмами
barcelona_map1_2019 = folium.Map(location=[41.38879, 2.15899], zoom_start=12)
for lat, lng, label in zip(df_2019.latitud, df_2019.longitud, df_2019.numero_lesionats_greus.astype(str)):
    if label != '0':
        folium.CircleMarker(
            [lat, lng],
            radius=3,
            color='red',
            fill=True,
            popup=label,
            fill_color='darkred',
            fill_opacity=0.6
        ).add_to(barcelona_map1_2019)
barcelona_map1_2019.save("map1_2019.html")

# Нещасні випадки на карті з серйозними травмами з групуванням
barcelona_map2_2019 = folium.Map(location=[41.38879, 2.15899], zoom_start=12)
accidents = plugins.MarkerCluster().add_to(barcelona_map2_2019)
for lat, lng, label in zip(df_2019.latitud, df_2019.longitud, df_2019.numero_lesionats_greus.astype(str)):
    if label != '0':
        folium.Marker(
            location=[lat, lng],
            icon=None,
            popup=label,
        ).add_to(accidents)
barcelona_map2_2019.save("map2_2019.html")

# Інтерактивна групова карта
barcelona_map3_2019 = folium.Map(location=[41.38879, 2.15899], zoom_start=12)
hour_list = [[] for _ in range(24)]
for lat, log, hour in zip(df_2019.latitud, df_2019.longitud, df_2019.date.dt.hour):
    hour_list[hour].append([lat, log])
index = [str(i) + ' Hours' for i in range(24)]
HeatMapWithTime(hour_list, index).add_to(barcelona_map3_2019)
barcelona_map3_2019.save("map3_2019.html")

# Усі автокатастрофи по порі доби
barcelona_map4_1_2019 = folium.Map(location=[41.38879, 2.15899], zoom_start=12)
for lat, lng, label, hour in zip(df_2019.latitud, df_2019.longitud, df_2019.numero_victimes.astype(str),
                                 df_2019['date'].dt.hour):
    if label != '0' and hour > 21:
        folium.CircleMarker(
            [lat, lng],
            radius=3,
            color='red',
            fill=True,
            popup=label,
            fill_color='darkred',
            fill_opacity=0.6
        ).add_to(barcelona_map4_1_2019)
    if label != '0' and 21 > hour > 14:
        folium.CircleMarker(
            [lat, lng],
            radius=3,
            color='blue',
            fill=True,
            popup=label,
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(barcelona_map4_1_2019)
    if label != '0' and hour < 14:
        folium.CircleMarker(
            [lat, lng],
            radius=3,
            color='green',
            fill=True,
            popup=label,
            fill_color='green',
            fill_opacity=0.6
        ).add_to(barcelona_map4_1_2019)
barcelona_map4_1_2019.save("map4_1_2019.html")

# Автокатастрофи з тяжкими наслідками по порі доби
barcelona_map4_2_2019 = folium.Map(location=[41.38879, 2.15899], zoom_start=12)
for lat, lng, label, hour in zip(df_2019.latitud, df_2019.longitud, df_2019.numero_lesionats_greus.astype(str),
                                 df_2019['date'].dt.hour):
    if label != '0' and hour > 21:
        folium.CircleMarker(
            [lat, lng],
            radius=3,
            color='red',
            fill=True,
            popup=label,
            fill_color='darkred',
            fill_opacity=0.6
        ).add_to(barcelona_map4_2_2019)
    if label != '0' and 21 > hour > 14:
        folium.CircleMarker(
            [lat, lng],
            radius=3,
            color='blue',
            fill=True,
            popup=label,
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(barcelona_map4_2_2019)
    if label != '0' and hour < 14:
        folium.CircleMarker(
            [lat, lng],
            radius=3,
            color='green',
            fill=True,
            popup=label,
            fill_color='green',
            fill_opacity=0.6
        ).add_to(barcelona_map4_2_2019)
barcelona_map4_2_2019.save("map4_2_2019.html")