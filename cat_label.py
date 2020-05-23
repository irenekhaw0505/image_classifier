import json

def load_json_data():
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)
    return cat_to_name


