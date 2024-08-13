from collections import defaultdict

# Initialize a defaultdict with lists as default values
logs = defaultdict(list)

# Adding some data
logs['info'].append('This is an info message.')
logs['error'].append('This is an error message.')
logs['warning'].append('This is a warning message.')

# Accessing the data
print(logs['info'])    # Output: ['This is an info message.']
print(logs['error'])   # Output: ['This is an error message.']
print(logs['warning']) # Output: ['This is a warning message.']

# Accessing a non-existent key returns an empty list
print(logs['debug'])   # Output: []
