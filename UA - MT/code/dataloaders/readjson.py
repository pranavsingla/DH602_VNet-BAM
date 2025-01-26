import json

base_path = '/home/pranav/DH602_VNet-BAM/dataset/MSD-spleen/Spleen/'
# Load the JSON data from a file
with open(f'{base_path}dataset.json', 'r') as f:
    data = json.load(f)

# Extract train and test lists
train_list = [entry['image'] for entry in data['training']]
test_list = data['test']

train_list += test_list
test_list = train_list[ : int(len(train_list) * 0.2)]
train_list = train_list[int(len(train_list) * 0.2) : ]
# test_list = test_list[ : int((len(test_list) + len(train_list)) * 0.2) - 1]

# Save the train list to a text file
with open(f'{base_path}train.list', 'w') as f:
    for item in train_list:
        f.write(f"{item}\n")

# Save the test list to a text file
with open(f'{base_path}test.list', 'w') as f:
    for item in test_list:
        f.write(f"{item}\n")

print("train.list and test.list have been created successfully.")

# import json

# base_path = '/home/pranav/DH602_VNet-BAM/dataset/MSD-spleen/Spleen/'
# # Load the JSON data from a file
# with open(f'{base_path}dataset.json', 'r') as f:
#     data = json.load(f)

# # Extract train and test lists
# train_list = [entry['image'] for entry in data['training']]
# test_list = data['test']

# train_list += test_list[int((len(test_list) + len(train_list)) * 0.2) : ]
# test_list = test_list[ : int((len(test_list) + len(train_list)) * 0.2) - 1]

# # Save the train list to a text file
# with open(f'{base_path}train.list', 'w') as f:
#     for item in train_list:
#         f.write(f"{item}\n")

# # Save the test list to a text file
# with open(f'{base_path}test.list', 'w') as f:
#     for item in test_list:
#         f.write(f"{item}\n")

# print("train.list and test.list have been created successfully.")
