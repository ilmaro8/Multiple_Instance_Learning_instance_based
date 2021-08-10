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

import warnings
warnings.filterwarnings("ignore")

args = sys.argv[1:]

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
parser.add_argument('-w', '--wsi_folder', help='path where WSIs are stored',type=str, default='./images/')

args = parser.parse_args()

CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
pool_algorithm = args.pool
TASK = args.TASK
EMBEDDING_bool = args.features
INPUT_DATA = args.input
MODEL_PATH = args.model
WSI_FOLDER = args.wsi_folder

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

def generate_list_instances(filename):

	instance_dir = WSI_FOLDER
	fname = os.path.split(filename)[-1]
	
	instance_csv = instance_dir+fname+'/'+fname+'_paths_densely.csv'

	return instance_csv 

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

		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)
		
		self.fc_feat_in = fc_input_features
		self.N_CLASSES = N_CLASSES
		
		if (EMBEDDING_bool==True):

			if ('resnet34' in CNN_TO_USE):
				self.E = 128
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES

			elif ('resnet50' in CNN_TO_USE):
				self.E = 256
				self.L = self.E
				self.D = 128
				self.K = self.N_CLASSES

			
			self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
			self.embedding_fc = torch.nn.Linear(in_features=self.E, out_features=self.N_CLASSES)

		else:
			self.fc = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.N_CLASSES)

			if ('resnet34' in CNN_TO_USE):
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
			embedding_layer = self.embedding(conv_layers_out)
			output_fcn = self.embedding_fc(embedding_layer)

			features_to_return = embedding_layer
		else:
			output_fcn = self.fc(conv_layers_out)
			features_to_return = conv_layers_out

		#print(output_fcn.shape)
		if (TASK=='binary' and N_CLASSES==1):
			output_fcn = m_binary(output_fcn)
		else:
			output_fcn = m_multiclass(output_fcn)

		output_fcn = torch.clamp(output_fcn, 1e-7, 1 - 1e-7)
		#print(output_fcn.size())

		if (pool_algorithm=='max'):
			output_pool = output_fcn.max(dim = 0)[0]
		elif (pool_algorithm=='avg'):
			output_pool = output_fcn.mean(dim = 0)
			#print(output_pool.size())
		elif (pool_algorithm=='lin'):
			output_pool = (output_fcn * output_fcn).sum(dim = 0) / output_fcn.sum(dim = 0)
			#print(output_pool.size())
		elif (pool_algorithm=='exp'):
			output_pool = (output_fcn * output_fcn.exp()).sum(dim = 0) / output_fcn.exp().sum(dim = 0)
			#print(output_pool.size())
		elif (pool_algorithm=='att'):

			if (EMBEDDING_bool==True):
				A = self.attention(embedding_layer)
			else:
				A = self.attention(conv_layers_out)  # NxK
			"""
			A = torch.transpose(A, 1, 0)
			A = F.softmax(A, dim=1)  # softmax over N
			A = A.view(-1, A.size()[0])
			"""
			#A = torch.transpose(A, 1, 0)
			A = F.softmax(A, dim=0)  # softmax over N
			#A = A.view(-1, A.size()[0])
			#print(A)

			output_pool = (output_fcn * A).sum(dim = 0) / (A).sum(dim = 0)
			output_pool = torch.clamp(output_pool, 1e-7, 1 - 1e-7)
			#print(output_pool.size())

		return output_pool, output_fcn, A, features_to_return

def accuracy_micro(y_true, y_pred):

    y_true_flatten = y_true.flatten()
    y_pred_flatten = y_pred.flatten()
    
    return metrics.accuracy_score(y_true_flatten, y_pred_flatten)

    
def accuracy_macro(y_true, y_pred):
    
    n_classes = len(y_true[0])
    
    acc_tot = 0.0
    
    for i in range(n_classes):
        
        acc = metrics.accuracy_score(y_true[:,i], y_pred[:,i])
        #print(acc)
        acc_tot = acc_tot + acc
        
    acc_tot = acc_tot/n_classes
    
    return acc_tot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision import transforms
preprocess = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, partition):
		self.list_IDs = list_IDs
		self.set = partition

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index][0]
		# Load data and get label
		img = Image.open(ID)
		X = np.asarray(img)
		img.close()
		#data transformation
		input_tensor = preprocess(X).type(torch.FloatTensor)
				
		#return input_tensor
		return input_tensor
	
class Dataset_bag(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs
		
	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]
		
		# Load data and get label
		instances_filename = generate_list_instances(ID)
		y = self.labels[index]
		if (TASK=='binary' and N_CLASSES==1):
			y = np.asarray(y)
		else:
			y = torch.tensor(y.tolist() , dtype=torch.float32)

				
		return instances_filename, y

batch_size_bag = 1

params_test_bag = {'batch_size': batch_size_bag,
		  'shuffle': True}

if (TASK=='binary' and N_CLASSES==1):
	testing_set_bag = Dataset_bag(test_dataset[:,0], test_dataset[:,1])
	testing_generator_bag = data.DataLoader(testing_set_bag, **params_test_bag)
else:
	testing_set_bag = Dataset_bag(test_dataset[:,0], test_dataset[:,1:])
	testing_generator_bag = data.DataLoader(testing_set_bag, **params_test_bag)

print("testing")
print("testing at WSI level")
y_pred = []
y_true = []

model = torch.load(model_weights_filename)
model.to(device)
model.eval()

kappa_score_general_filename = checkpoint_path+'kappa_score_general_'+TASK+'.csv'
acc_balanced_filename = checkpoint_path+'acc_balanced_general_'+TASK+'.csv'
acc_filename = checkpoint_path+'acc_general_'+TASK+'.csv'
acc_macro_filename = checkpoint_path+'acc_macro_general_'+TASK+'.csv'
acc_micro_filename = checkpoint_path+'acc_micro_general_'+TASK+'.csv'
confusion_matrix_filename = checkpoint_path+'conf_matr_general_'+TASK+'.csv'
roc_auc_filename = checkpoint_path+'roc_auc_general_'+TASK+'.csv'
f1_score_macro_filename = checkpoint_path+'f1_macro_'+TASK+'.csv'
f1_score_micro_filename = checkpoint_path+'f1_micro_'+TASK+'.csv'
hamming_loss_filename = checkpoint_path+'hamming_loss_general_'+TASK+'.csv'
recall_score_macro_filename = checkpoint_path+'recall_score_macro_general_'+TASK+'.csv'
recall_score_micro_filename = checkpoint_path+'recall_score_micro_general_'+TASK+'.csv'
jaccard_score_macro_filename = checkpoint_path+'jaccard_score_macro_'+TASK+'.csv'
jaccard_score_micro_filename = checkpoint_path+'jaccard_score_micro_'+TASK+'.csv'
roc_auc_score_macro_filename = checkpoint_path+'roc_auc_score_macro_general_'+TASK+'.csv'
roc_auc_score_micro_filename = checkpoint_path+'roc_auc_score_micro_general_'+TASK+'.csv'
precision_score_macro_filename = checkpoint_path+'precision_score_macro_general_'+TASK+'.csv'
precision_score_micro_filename = checkpoint_path+'precision_score_micro_general_'+TASK+'.csv'
auc_score_filename = checkpoint_path+'auc_score_general_'+TASK+'.csv'

def save_metric(filename,value):
	array = [value]
	File = {'val':array}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename, df.values, fmt='%s',delimiter=',')

filenames_wsis = []
pred_cancers = []
pred_hgd = []
pred_lgd = []
pred_hyper = []
pred_normal = []

with torch.no_grad():
	j = 0
	for inputs_bag,labels in testing_generator_bag:
			#inputs: bags, labels: labels of the bags
		labels_np = labels.cpu().data.numpy()
		len_bag = len(labels_np)

			#list of bags 
		print("inputs_bag " + str(inputs_bag))

		filename_wsi = inputs_bag[0].split('/')[-2]

		inputs_bag = list(inputs_bag)

		try:

			for b in range(len_bag):
				labs = []
				labs.append(labels_np[b])
				labs = np.array(labs).flatten()

				labels = torch.tensor(labs).float().to(device)

					#read csv with instances
				csv_instances = pd.read_csv(inputs_bag[b], sep=',', header=None).values
					#number of instances
				n_elems = len(csv_instances)

					#params generator instances
				batch_size_instance = int(BATCH_SIZE_str)

				num_workers = 4
				params_instance = {'batch_size': batch_size_instance,
						'shuffle': True,
						'num_workers': num_workers}

					#generator for instances
				instances = Dataset_instance(csv_instances,'valid')
				validation_generator_instance = data.DataLoader(instances, **params_instance)
				
				features = []
				with torch.no_grad():
					for instances in validation_generator_instance:
						instances = instances.to(device)

						# forward + backward + optimize
						feats = model.conv_layers(instances)
						feats = feats.view(-1, fc_input_features)
						feats_np = feats.cpu().data.numpy()
						
						features = np.append(features,feats_np)
						
				#del instances

				features_np = np.reshape(features,(n_elems,fc_input_features))
				
				del features, feats
				
				inputs = torch.tensor(features_np).float().to(device)
				
				predictions, _, _, _ = model(None, inputs)

				outputs_np = predictions.cpu().data.numpy()
				labels_np = labels.cpu().data.numpy()

				filenames_wsis = np.append(filenames_wsis,filename_wsi)
				pred_cancers = np.append(pred_cancers,outputs_np[0])
				pred_hgd = np.append(pred_hgd,outputs_np[1])
				pred_lgd = np.append(pred_lgd,outputs_np[2])
				pred_hyper = np.append(pred_hyper,outputs_np[3])
				pred_normal = np.append(pred_normal,outputs_np[4])

				#print(outputs_np,labels_np)
				print("["+str(j)+"/"+str(len(test_dataset))+"]")
				print("output: "+str(outputs_np))
				print("ground truth:" + str(labels_np))
				outputs_np = np.where(outputs_np > 0.5, 1, 0)

				torch.cuda.empty_cache()

				y_pred = np.append(y_pred,outputs_np)
				y_true = np.append(y_true,labels_np)

				j = j + 1

		except:

			pass


filename_training_predictions = checkpoint_path+'WSI_predictions_AOEC.csv'

File = {'filenames':filenames_wsis, 'pred_cancers':pred_cancers, 'pred_hgd':pred_hgd,'pred_lgd':pred_lgd, 'pred_hyper':pred_hyper, 'pred_normal':pred_normal}

df = pd.DataFrame(File,columns=['filenames','pred_cancers','pred_hgd','pred_lgd','pred_hyper','pred_normal'])
np.savetxt(filename_training_predictions, df.values, fmt='%s',delimiter=',')	

y_pred = np.reshape(y_pred,(j,N_CLASSES))
y_true = np.reshape(y_true,(j,N_CLASSES))

try:
	accuracy_score = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
	print("accuracy_score : " + str(accuracy_score))
	save_metric(acc_filename,accuracy_score)
except:
	pass

try:
	accuracy_macro_score = accuracy_macro(y_true=y_true, y_pred=y_pred)
	print("accuracy_macro_score : " + str(accuracy_macro_score))
	save_metric(acc_macro_filename,accuracy_macro_score)
except:
	pass

try:
	accuracy_micro_score = accuracy_micro(y_true=y_true, y_pred=y_pred)
	print("accuracy_micro_score : " + str(accuracy_micro_score))
	save_metric(acc_micro_filename,accuracy_micro_score)
except:
	pass

try:
	hamming_loss = metrics.hamming_loss(y_true=y_true, y_pred=y_pred, sample_weight=None)
	print("hamming_loss : " + str(hamming_loss))
	save_metric(hamming_loss_filename,hamming_loss)
except:
	pass

try:
	zero_one_loss = metrics.zero_one_loss(y_true=y_true, y_pred=y_pred)
	print("zero_one_loss : " + str(zero_one_loss))
except:
	pass

try:
	multilabel_confusion_matrix = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred)
	print("multilabel_confusion_matrix: ")
	print(multilabel_confusion_matrix)
	save_metric(confusion_matrix_filename,multilabel_confusion_matrix)
except:
	pass

try:
	target_names = ['cancer', 'hgd', 'lgd', 'hyper']
	classification_report = metrics.classification_report(y_true, y_pred, target_names=target_names)
	print("classification_report: ")
	print(classification_report)
except:
	pass

try:
	jaccard_score_macro = metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average='macro')
	jaccard_score_micro = metrics.jaccard_score(y_true=y_true, y_pred=y_pred, average='micro')
	print("jaccard_score_macro : " + str(jaccard_score_macro))
	print("jaccard_score_micro : " + str(jaccard_score_micro))
	save_metric(jaccard_score_macro_filename,jaccard_score_macro)
	save_metric(jaccard_score_micro_filename,jaccard_score_micro)
except:
	pass

try:
	f1_score_macro = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro')
	f1_score_micro = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')
	print("f1_score_macro : " + str(f1_score_macro))
	print("f1_score_micro : " + str(f1_score_micro))
	save_metric(f1_score_macro_filename,f1_score_macro)
	save_metric(f1_score_micro_filename,f1_score_micro)
except:
	pass

try:
	recall_score_macro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='macro')
	recall_score_micro = metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro')
	print("recall_score_macro : " + str(recall_score_macro))
	print("recall_score_micro : " + str(recall_score_micro))
	save_metric(recall_score_macro_filename,recall_score_macro)
	save_metric(recall_score_micro_filename,recall_score_micro)
except:
	pass

try:
	precision_score_macro = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='macro')
	precision_score_micro = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro')
	print("precision_score_macro : " + str(precision_score_macro))
	print("precision_score_micro : " + str(precision_score_micro))
	save_metric(precision_score_macro_filename,precision_score_macro)
	save_metric(precision_score_micro_filename,precision_score_micro)
except:
	pass

try:
	roc_auc_score_macro = metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='macro')
	roc_auc_score_micro = metrics.roc_auc_score(y_true=y_true, y_score=y_pred, average='micro')
	print("roc_auc_score_macro : " + str(roc_auc_score_macro))
	print("roc_auc_score_micro : " + str(roc_auc_score_micro)) 
	save_metric(roc_auc_score_macro_filename,roc_auc_score_macro)
	save_metric(roc_auc_score_micro_filename,roc_auc_score_macro)
except:
	pass      
