import pandas as pd
import geopandas as gpd
import folium

data = pd.read_csv('data/data.csv')
geo_data = gpd.read_file("json/seoul.geojson")

data = data.dropna()
data['Veg density'] = data['Veg area (km2)'] / data['Area (km2)']
data = data[42:67] # Only the city of Seoul

data['district'] = data['Geography']
geo_data['district'] = geo_data['SIG_ENG_NM']


m = folium.Map(location=[37.5665, 126.9780], zoom_start=10)

folium.Choropleth(
    geo_data=geo_data,
    data=data,
    columns=['district', 'pm25 mcg/m3'],
    key_on='feature.properties.district',
    fill_color='YlOrBr',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='pm25 mcg/m3'
).add_to(m)

m.save("html/seoul_clusters_map.html")