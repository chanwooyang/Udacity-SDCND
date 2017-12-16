import os
import pickle
import time

from sklearn.svm import LinearSVC

def linearSVC_Classifier(train_features,train_labels,test_features,test_labels):

	if os.path.isfile('ClassifierData.p'):
		data_file = 'ClassifierData.p'
		with open(data_file, mode='rb') as f:
			data = pickle.load(f)
		svc = data['svc']
	else:
		svc = LinearSVC()
		print('Training Linear SVC...')
		t = time.time()
		svc.fit(train_features,train_labels)
		t1 = time.time()
		print('Training Complete. ',round(t1-t,2), ' seconds to train Linear SVC.')
		accuracy = svc.score(test_features,test_labels)
		print('Test Accuracy of Linear SVC: {0:.4f}'.format(accuracy))


		# Save svc as pickle
		pickle_file = 'ClassifierData.p'
		print('Saving data to pickle file...')
		try:
			with open(pickle_file, 'wb') as pfile:
				pickle.dump(
					{	'svc':svc
					},
					pfile, pickle.HIGHEST_PROTOCOL)
		except Exception as e:
			print('Unable to save data to', pickle_file, ':', e)
			raise

	return svc