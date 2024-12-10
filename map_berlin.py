import pandas as pd
import geopandas as gpd
import folium

data = pd.read_csv('data/data.csv')
geo_data = gpd.read_file("json/berlin-neighbourhoods.geojson")

data = data.dropna()
data['Veg density'] = data['Veg area (km2)'] / data['Area (km2)']
data = data[67:]

# Adding a colum 'neighbourhood_group' in the DataFrame to merge it with the GeoDataFrame
data['neighbourhood_group'] = data['Geography'].str[:3]
# Same name needed
geo_data['neighbourhood_group'] = geo_data['neighbourhood_group'].str[:3]

# Unifiying the groups of neighbourhoods
grouped_geo_data = geo_data.dissolve(by='neighbourhood_group')

m = folium.Map(location=[52.52, 13.405], zoom_start=10)

folium.Choropleth(
    geo_data=grouped_geo_data.to_json(),
    data=data,
    columns=['neighbourhood_group', 'Area (km2)'],
    key_on='id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Area (km2)'
).add_to(m)

m.save("berlin_groups_map.html")