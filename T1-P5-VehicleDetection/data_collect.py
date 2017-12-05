import os


dataset_path = 'dataset'
if os.path.exists(dataset_path):
	pass
else:
	raise TypeError("Dataset directory does not exist")