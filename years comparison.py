import pandas as pd
import plotly.graph_objs as go
from folium import plugins
from folium.plugins import HeatMapWithTime
import folium
import seaborn as sns


df_2017 = pd.read_csv('2017_accidents.csv')
df_2018 = pd.read_csv('2018_accidents.csv')
df_2019 = pd.read_csv('2019_accidents.csv')

# Виявлення та видалення нульових значень
df_2018.drop_duplicates(inplace=True)
df_2019.drop_duplicates(inplace=True)
df_2017.drop_duplicates(inplace=True)

df_2017.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
print(df_2017.columns)
df_2018.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
print(df_2018.columns)
df_2019.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
print(df_2019.columns)

Index = [df_2017.numero_vehicles_implicats.count(), df_2018.numero_vehicles_implicats.count(),
         df_2019.numero_vehicles_implicats.count()]

# Співвідношення кількості ДТП
fig = go.Figure(data=[go.Bar(x=[2017, 2018, 2019], y=Index,
                             marker_color='rgb(192,192,192)', marker_line_color='rgb(0,0,0)',
                             marker_line_width=1.5, opacity=0.6, width=0.5,
                             text=Index, textposition='outside', textfont_size=20)])
fig.update_layout(title='ДТП в Барселоні', titlefont=dict(size=25),
                  yaxis_title='Кількість аварій в році', xaxis_title='Рік')
fig.show()

# Співвідношення кількості автомобілів
vehicles_involved = [df_2017.numero_vehicles_implicats.sum(), df_2018.numero_vehicles_implicats.sum(),
                     df_2019.numero_vehicles_implicats.sum()]
fig5 = go.Figure(data=[
    go.Bar(x = [2017,2018,2019], y=vehicles_involved, marker_color='rgb(152,251,152)', marker_line_color='rgb(46,139,87)',
                             marker_line_width=1.5, opacity=0.6, width=0.5,
                             text=vehicles_involved, textposition='outside', textfont_size=20)])
fig5.update_layout(title='Задіяний транспортний засіб', titlefont=dict(size=25),
                  yaxis_title='Кількість задіяних транспортних засобів', xaxis_title='Рік')
fig5.show()

# Співвідношення кількості смертей
serious_injuries = [df_2017.numero_lesionats_greus.sum(), df_2018.numero_lesionats_greus.sum(),
                    df_2019.numero_lesionats_greus.sum()]
fig5 = go.Figure(data=[
    go.Bar(x = [2017,2018,2019], y=serious_injuries, marker_color='rgb(152,251,152)', marker_line_color='rgb(46,139,87)',
                             marker_line_width=1.5, opacity=0.6, width=0.5,
                             text=serious_injuries, textposition='outside', textfont_size=20)])
fig5.update_layout(title='Тяжкі нещасні випадки', titlefont=dict(size=25),
                  yaxis_title='Кількість тяжких випадків', xaxis_title='Рік')
fig5.show()


latitud = pd.concat([df_2017.latitud, df_2018.latitud, df_2019.latitud], ignore_index=True)
longitude = pd.concat([df_2017.longitud, df_2018.longitud, df_2019.longitud], ignore_index=True)
serious_injuries = pd.concat([df_2017.numero_lesionats_greus.astype(str), df_2018.numero_lesionats_greus.astype(str),
                              df_2019.numero_lesionats_greus.astype(str)], ignore_index=True)

# Сумарна інтерактивна карта кількості важких випадків
barcelona_map_sum = folium.Map(location=[41.38879, 2.15899], zoom_start=12)
accidents = plugins.MarkerCluster().add_to(barcelona_map_sum)
for lat, lng, label in zip(latitud,
                           longitude,
                           serious_injuries):
    if label != '0':
        folium.Marker(
            location=[lat, lng],
            icon=None,
            popup=label,
        ).add_to(accidents)
barcelona_map_sum.save("map_sum.html")

hour = pd.concat([df_2017.hora_dia, df_2018.hora_dia,
                  df_2019.hora_dia], ignore_index=True)

# Сумарна інтерактивна карта по годинах
barcelona_map_sum_hour = folium.Map(location=[41.38879, 2.15899], zoom_start=12)
hour_list = [[] for _ in range(24)]
for lat, log, hour in zip(latitud,
                          longitude,
                          hour):
    hour_list[hour].append([lat, log])
index = [str(i) + ' Hours' for i in range(24)]
HeatMapWithTime(hour_list, index).add_to(barcelona_map_sum_hour)
barcelona_map_sum_hour.save("map_sum_hour.html")

