import sys, getopt
import torch
from torch.utils import data
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.utils.data
from sklearn import metrics 
import os
import argparse

args = sys.argv[1:]

import warnings
warnings.filterwarnings("ignore")

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-c', '--CNN', help='cnn architecture to use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=512)
parser.add_argument('-p', '--pool', help='pooling algorithm',type=str, default='att')
parser.add_argument('-t', '--TASK', help='task (binary/multilabel)',type=str, default='resnet34')
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=bool, default=True)
parser.add_argument('-m', '--model', help='path of the model to load',type=str, default='./model/')
parser.add_argument('-i', '--input', help='path of input csv',type=str, default='./model/')

args = parser.parse_args()

CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
pool_algorithm = args.pool
TASK = args.TASK
EMBEDDING_bool = args.features
INPUT_DATA = args.input
MODEL_PATH = args.model


print("PARAMETERS")
print("TASK: " + str(TASK))
print("CNN used: " + str(CNN_TO_USE))
print("POOLING ALGORITHM: " + str(pool_algorithm))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))

#create folder (used for saving weights)
def create_dir(models_path):
	if not os.path.isdir(models_path):
		try:
			os.mkdir(models_path)
		except OSError:
			print ("Creation of the directory %s failed" % models_path)
		else:
			print ("Successfully created the directory %s " % models_path)

#DIRECTORIES CREATION

checkpoint_path = MODEL_PATH+'checkpoints_MIL/'
create_dir(checkpoint_path)

#path model file
model_weights_filename = MODEL_PATH


print("CSV LOADING ")
csv_filename_testing = INPUT_DATA


#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

#MODEL DEFINITION
pre_trained_network = torch.hub.load('pytorch/vision:v0.4.2', CNN_TO_USE, pretrained=True)

if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.fc.in_features
elif (('densenet' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.classifier.in_features
elif ('mobilenet' in CNN_TO_USE):
	fc_input_features = pre_trained_network.classifier[1].in_features

class MIL_model(torch.nn.Module):
	def __init__(self):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(MIL_model, self).__init__()
		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])
		#self.conv_layers = siamese_model.conv_layers
		"""
		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)
		"""
		self.fc_feat_in = fc_input_features
		self.N_CLASSES = N_CLASSES
		
		if (EMBEDDING_bool==True):

			if ('resnet18' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES

			elif ('resnet34' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES
				#self.K = 1
			elif ('resnet50' in CNN_TO_USE):
				self.E = 256
				self.L = self.E
				self.D = 128
				self.K = self.N_CLASSES

			#self.embedding = siamese_model.embedding
			self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
			self.embedding_fc = torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES)

		else:
			self.fc = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.N_CLASSES)
			
			if ('resnet18' in CNN_TO_USE):
				self.L = fc_input_features
				self.D = 128
				self.K = self.N_CLASSES

			elif ('resnet34' in CNN_TO_USE):
				self.L = fc_input_features
				self.D = 128
				self.K = self.N_CLASSES

			elif ('resnet50' in CNN_TO_USE):
				self.L = self.E
				self.D = 256
				self.K = self.N_CLASSES

		if (pool_algorithm=='att'):

			self.attention = torch.nn.Sequential(
				torch.nn.Linear(self.L, self.D),
				torch.nn.Tanh(),
				torch.nn.Linear(self.D, self.K)
			)

		self.tanh = torch.nn.Tanh()
		self.relu = torch.nn.ReLU()

	def forward(self, x, conv_layers_out):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		#if used attention pooling
		A = None
		#m = torch.nn.Softmax(dim=1)
		m_binary = torch.nn.Sigmoid()
		m_multiclass = torch.nn.Softmax()
		dropout = torch.nn.Dropout(p=0.2)
		
		
		if x is not None:
			#print(x.shape)
			conv_layers_out=self.conv_layers(x)
			#print(x.shape)
			
			conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

		#print(conv_layers_out.shape)

		if ('mobilenet' in CNN_TO_USE):
			dropout = torch.nn.Dropout(p=0.2)
			conv_layers_out = dropout(conv_layers_out)
		#print(conv_layers_out.shape)

		if (EMBEDDING_bool==True):
			#conv_layers_out = self.tanh(conv_layers_out)
			embedding_layer = self.embedding(conv_layers_out)
			#embedding_layer = self.tanh(embedding_layer)
			embedding_layer = self.relu(embedding_layer)
			features_to_return = embedding_layer

			#embedding_layer = self.tanh(embedding_layer)

			embedding_layer = dropout(embedding_layer)
			logits = self.embedding_fc(embedding_layer)

		else:
			logits = self.fc(conv_layers_out)
			features_to_return = conv_layers_out
		#print(output_fcn.shape)
		

		#print(output_fcn.size())

		if (EMBEDDING_bool==True):
			A = self.attention(features_to_return)
		else:
			A = self.attention(conv_layers_out)  # NxK
			
		#print(A.size())
		#print(A)
		A = F.softmax(A, dim=0)  # softmax over N
		#print(A.size())
		#print(A)
		#A = A.view(-1, A.size()[0])
		#print(A)

		output_pool = (logits * A).sum(dim = 0) #/ (A).sum(dim = 0)
		#print(output_pool.size())
		#print(output_pool)
		#output_pool = torch.clamp(output_pool, 1e-7, 1 - 1e-7)

		output_fcn = m_multiclass(logits)
		return output_pool, output_fcn, A, features_to_return


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(model_weights_filename)
model.to(device)
model.eval()

from torchvision import transforms
preprocess = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset_test_strong(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):

		# Select sample
		ID = self.list_IDs[index]
		# Load data and get label
		X = Image.open(ID)
		X = np.asarray(X)
		y = self.labels[index]
		#data augmentation
		#geometrical

		#data transformation
		input_tensor = preprocess(X).type(torch.FloatTensor)
				
		return input_tensor, np.asarray([y]), ID

#read data
test_dataset = pd.read_csv(csv_filename_testing, sep=',', header=None).values

params_test = {'batch_size': int(BATCH_SIZE_str),
		  #'shuffle': True,
		  #'sampler': ImbalancedDatasetSampler(train_dataset),
		  'num_workers': 2}

testing_set_strong = Dataset_test_strong(test_dataset[:,0], test_dataset[:,1])
testing_generator_strong = data.DataLoader(testing_set_strong, **params_test)

y_pred = []
y_true = []

filenames = []
outputs_store = []
cumulative_labels = []

with torch.no_grad():
	for inputs, labels, filename in testing_generator_strong:
		inputs, labels = inputs.to(device), labels.type(torch.FloatTensor).to(device)
		
		_, outputs, _, _ = model(inputs, None) 

		#accumulate values
		outputs_np = outputs.cpu().data.numpy()
		labels_np = labels.cpu().data.numpy()

		filenames = np.append(filenames,filename)
		outputs_store = np.append(outputs_store,outputs_np)
		cumulative_labels = np.append(cumulative_labels,labels_np)

		outputs_np = np.argmax(outputs_np, axis=1)
			   
		y_pred = np.append(y_pred,outputs_np)
		y_true = np.append(y_true,labels_np)

#k-score
k_score = metrics.cohen_kappa_score(y_true,y_pred, weights='quadratic')
print("k_score " + str(k_score))
#f1_scrre
f1_score = metrics.f1_score(y_true, y_pred, average='macro')
print("f1_score " + str(f1_score))
#confusion matrix
confusion_matrix = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
print("confusion_matrix ")
print(str(confusion_matrix))
acc_balanced = metrics.balanced_accuracy_score(y_true, y_pred, sample_weight=None, adjusted=False)
print("acc_balanced " + str(acc_balanced))
try:
	roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
	print("roc_auc " + str(roc_auc_score))
except:
	pass


kappa_score_general_filename = checkpoint_path+'kappa_score_general_multiclass_strong.csv'
acc_balanced_filename = checkpoint_path+'acc_balanced_general_multiclass_strong.csv'
confusion_matrix_filename = checkpoint_path+'conf_matr_general_multiclass_strong.csv'
roc_auc_score_filename = checkpoint_path+'roc_auc_score_general_multiclass_strong.csv'
f1_score_filename = checkpoint_path+'f1_score_general_multiclass_strong.csv'

kappas = [k_score]

File = {'val':kappas}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(kappa_score_general_filename, df.values, fmt='%s',delimiter=',')

f1_scores = [f1_score]

File = {'val':f1_scores}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(f1_score_filename, df.values, fmt='%s',delimiter=',')

acc_balancs = [acc_balanced]

File = {'val':acc_balancs}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(acc_balanced_filename, df.values, fmt='%s',delimiter=',')

conf_matr = [confusion_matrix]
File = {'val':conf_matr}
df = pd.DataFrame(File,columns=['val'])
np.savetxt(confusion_matrix_filename, df.values, fmt='%s',delimiter=',')

try:
	roc_auc = [roc_auc_score]
	File = {'val':roc_auc}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(roc_auc_score_filename, df.values, fmt='%s',delimiter=',')
except:
	pass

predictions_radboudc_filename = checkpoint_path+'predictions_raw_aoec.csv'

outputs_store = np.reshape(outputs_store,(len(test_dataset),5))

#print(outputs_store)

pred_cancer = [row[0] for row in outputs_store]
pred_hgd = [row[1] for row in outputs_store]
pred_lgd = [row[2] for row in outputs_store]
pred_hyper = [row[3] for row in outputs_store]
pred_normal = [row[4] for row in outputs_store]

File = {'filenames':filenames,'labels':cumulative_labels,'cancer':pred_cancer,'HGD':pred_hgd,'LGD':pred_lgd,'Hyper':pred_hyper,'Normal':pred_normal}
df = pd.DataFrame(File,columns=['filenames','labels','cancer','HGD','LGD','Hyper','Normal'])
np.savetxt(predictions_radboudc_filename, df.values, fmt='%s',delimiter=',')