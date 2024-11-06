import yaml
from functions import print_dict

# Read YAML file
with open("file_input.yaml", 'r') as file:
    data = yaml.safe_load(file)


# print(data)
print_dict(data)
# 