import pandas as pd
import seaborn as sns
import folium
import geopandas as gpd

data = pd.read_csv('data/data.csv')
data = data.dropna()  
data['Veg density'] = data['Veg area (km2)'] / data['Area (km2)']
data = data[67:]

# Charger les données géographiques (exemple avec GeoJSON pour Berlin)
geo_data = gpd.read_file("json/berlin-neighbourhoods.geojson")
geo_data['neighbourhood_group'] = geo_data['neighbourhood_group'].str[:3]
data['neighbourhood_group'] = data['Geography'].str[:3]
geo_data = geo_data.merge(data, on="neighbourhood_group")

print(data)
print(geo_data['neighbourhood_group'])
# Créer une carte interactive
m = folium.Map(location=[52.52, 13.405], zoom_start=10)

# Ajouter des clusters à la carte
folium.Choropleth(
    geo_data=geo_data,
    data=data,
    columns=['neighbourhood_group', 'Area (km2)'],
    key_on='feature.properties.neighbourhood_group',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Area (km2)'
).add_to(m)

m.save("berlin_clusters_map.html")
