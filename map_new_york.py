import pandas as pd
import geopandas as gpd
import folium

data = pd.read_csv('data/data.csv')
geo_data = gpd.read_file("json/nyc.json")

data = data.dropna()
data['Veg density'] = data['Veg area (km2)'] / data['Area (km2)']
data = data[:42] 

data['district'] = data['Geography']
geo_data['district'] = geo_data['GEONAME']

m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)

folium.Choropleth(
    geo_data=geo_data,
    data=data,
    columns=['district', 'pm25 mcg/m3'],
    key_on='feature.properties.GEONAME',
    fill_color='YlOrBr',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='pm25 mcg/m3'
).add_to(m)

m.save("html/nyc_clusters_map.html")

