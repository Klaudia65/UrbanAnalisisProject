import json

with open('data/air_pollution_raw.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

filtered_locations = {}
for val in data["DATA"]:
    if val["msrste_nm"].endswith("구"):
        filtered_locations[val["msrste_nm"]] =  val["no2"]

print(len(filtered_locations))