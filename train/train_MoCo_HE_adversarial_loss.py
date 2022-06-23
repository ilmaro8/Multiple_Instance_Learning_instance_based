import numpy as np
import pandas as pd
import os
from PIL import Image
import albumentations as A
import torch
from torch.utils import data
import torch.utils.data
import argparse
import warnings
import sys
from torch.utils.data import SubsetRandomSampler, WeightedRandomSampler, Sampler

from lars import LARS

warnings.filterwarnings("ignore")

argv = sys.argv[1:]

print("CUDA current device " + str(torch.cuda.current_device()))
print("CUDA devices available " + str(torch.cuda.device_count()))

#parser parameters
parser = argparse.ArgumentParser(description='Configurations to train models.')
parser.add_argument('-n', '--N_EXP', help='number of experiment',type=int, default=0)
parser.add_argument('-c', '--CNN', help='cnn_to_use',type=str, default='resnet34')
parser.add_argument('-b', '--BATCH_SIZE', help='batch_size',type=int, default=256)
parser.add_argument('-e', '--EPOCHS', help='epochs to train',type=int, default=10)
parser.add_argument('-m', '--MAG', help='magnification to select',type=str, default='10')
parser.add_argument('-f', '--features', help='features_to_use: embedding (True) or features from CNN (False)',type=str, default='True')
parser.add_argument('-l', '--lr', help='learning rate',type=float, default=1e-4)
parser.add_argument('-i', '--input_folder', help='path of the folder where train.csv and valid.csv are stored',type=str, default='./partition/')
parser.add_argument('-o', '--output_folder', help='path where to store the model weights',type=str, default='./models/')
parser.add_argument('-w', '--wsi_folder', help='path where WSIs are stored',type=str, default='./images/')

args = parser.parse_args()

N_EXP = args.N_EXP
N_EXP_str = str(N_EXP)
CNN_TO_USE = args.CNN
BATCH_SIZE = args.BATCH_SIZE
BATCH_SIZE_str = str(BATCH_SIZE)
EPOCHS = args.EPOCHS
EPOCHS_str = EPOCHS
MAGNIFICATION = args.MAG
EMBEDDING_bool = args.features
lr = args.lr
INPUT_FOLDER = args.input_folder
OUTPUT_FOLDER = args.output_folder
WSI_FOLDER = args.wsi_folder

if (EMBEDDING_bool=='True'):
	EMBEDDING_bool = True
else:
	EMBEDDING_bool = False
	
num_keys = 4096
num_keys = 8192
num_keys = 16384
num_keys = 32768
#num_keys = 65536


#print(EMBEDDING_bool)

seed = N_EXP
torch.manual_seed(seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

print("PARAMETERS")
print("N_EPOCHS: " + str(EPOCHS_str))
print("CNN used: " + str(CNN_TO_USE))
print("BATCH_SIZE: " + str(BATCH_SIZE_str))
print("MAGNIFICATION: " + str(MAGNIFICATION))

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
print("CREATING/CHECKING DIRECTORIES")

create_dir(OUTPUT_FOLDER)

models_path = OUTPUT_FOLDER
checkpoint_path = models_path+'checkpoints_MIL/'
create_dir(checkpoint_path)

#path model file
model_weights_filename = models_path+'MIL_colon_'+TASK+'.pt'
model_weights_filename_temporary = models_path+'MIL_colon_'+TASK+'_temporary.pt'


#CSV LOADING
print("CSV LOADING ")

k = 10
N_CLASSES = 5
csv_folder = INPUT_FOLDER

if (TASK=='binary'):

	N_CLASSES = 1
	#N_CLASSES = 2

	if (N_CLASSES==1):
		csv_filename_training = csv_folder+'train_binary.csv'
		csv_filename_validation = csv_folder+'valid_binary.csv'

	
elif (TASK=='multilabel'):

	N_CLASSES = 5

	csv_filename_training = csv_folder+'train_multilabel.csv'
	csv_filename_validation = csv_folder+'valid_multilabel.csv'

#read data
train_dataset = pd.read_csv(csv_filename_training, sep=',', header=None).values#[:10]
valid_dataset = pd.read_csv(csv_filename_validation, sep=',', header=None).values#[:10]

print(len(train_dataset))

n_centers = 1

#reverse autograd
from torch.autograd import Function
class ReverseLayerF(Function):

	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha

		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha

		return output, None

class domain_predictor(torch.nn.Module):
	def __init__(self, n_centers):
		super(domain_predictor, self).__init__()
		# domain predictor
		self.fc_feat_in = fc_input_features
		self.n_centers = n_centers
			
		if ('resnet18' in CNN_TO_USE):
			self.E = 128

		elif ('resnet34' in CNN_TO_USE):
			self.E = 128

		elif ('resnet50' in CNN_TO_USE):
			self.E = 256
		
		elif ('densenet121' in CNN_TO_USE):
			self.E = 128
			
		
		self.domain_embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
		self.domain_classifier = torch.nn.Linear(in_features=self.E, out_features=self.n_centers)

		#self.domain_predictor = domain_predictor(6)
		self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25) 

	def forward(self, x):

		dropout = torch.nn.Dropout(p=0.1)
		m_binary = torch.nn.Sigmoid()
		relu = torch.nn.ReLU()

		domain_emb = self.domain_embedding(x)

		domain_emb = self.prelu(domain_emb)
		domain_emb = dropout(domain_emb)

		domain_prob = self.domain_classifier(domain_emb)

		#domain_prob = m_binary(domain_prob)

		return domain_prob

pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)

if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.fc.in_features
elif (('densenet' in CNN_TO_USE)):
	fc_input_features = pre_trained_network.classifier.in_features
elif ('mobilenet' in CNN_TO_USE):
	fc_input_features = pre_trained_network.classifier[1].in_features


class Encoder(torch.nn.Module):
	def __init__(self, dim):
		"""
		In the constructor we instantiate two nn.Linear modules and assign them as
		member variables.
		"""
		super(Encoder, self).__init__()

		pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', CNN_TO_USE, pretrained=True)

		if (('resnet' in CNN_TO_USE) or ('resnext' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.fc.in_features
		elif (('densenet' in CNN_TO_USE)):
			fc_input_features = pre_trained_network.classifier.in_features
		elif ('mobilenet' in CNN_TO_USE):
			fc_input_features = pre_trained_network.classifier[1].in_features

		self.conv_layers = torch.nn.Sequential(*list(pre_trained_network.children())[:-1])

		if (torch.cuda.device_count()>1):
			self.conv_layers = torch.nn.DataParallel(self.conv_layers)

		self.fc_feat_in = fc_input_features
		self.N_CLASSES = N_CLASSES
		
		self.dim = dim

		if (EMBEDDING_bool==True):

			if ('resnet34' in CNN_TO_USE):
				self.E = self.dim
				self.L = self.E
				self.D = 64
				self.K = self.N_CLASSES

			elif ('resnet50' in CNN_TO_USE):
				self.E = self.dim
				self.L = self.E
				self.D = 128
				self.K = self.N_CLASSES

			elif ('resnet152' in CNN_TO_USE):
				self.E = self.dim
				self.L = self.E
				self.D = 128
				self.K = self.N_CLASSES


			self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)

		
		self.domain_predictor = domain_predictor(6)
		self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25) 

	def forward(self, x, mode, alpha):
		"""
		In the forward function we accept a Tensor of input data and we must return
		a Tensor of output data. We can use Modules defined in the constructor as
		well as arbitrary operators on Tensors.
		"""
		#if used attention pooling
		A = None
		#m = torch.nn.Softmax(dim=1)
		dropout = torch.nn.Dropout(p=0.2)
		relu = torch.nn.ReLU()
		tanh = torch.nn.Tanh()
		

		if x is not None:
			#print(x.shape)
			conv_layers_out=self.conv_layers(x)
			#print(x.shape)

			conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

		#print(conv_layers_out.shape)

		if ('mobilenet' in CNN_TO_USE):
			#dropout = torch.nn.Dropout(p=0.2)
			conv_layers_out = dropout(conv_layers_out)
		#print(conv_layers_out.shape)

		if (EMBEDDING_bool==True):
			#conv_layers_out = relu(conv_layers_out)
			#conv_layers_out = dropout(conv_layers_out)
			embedding_layer = self.embedding(conv_layers_out)
			embedding_layer = self.prelu(embedding_layer)
			
			features_to_return = embedding_layer

		else:
			features_to_return = conv_layers_out

		norm = torch.norm(features_to_return, p='fro', dim=1, keepdim=True)

		#normalized_array = features_to_return #/ norm
		#normalized_array = features_to_return 
		normalized_array = torch.nn.functional.normalize(features_to_return, dim=1)

		if (mode=='train'):
			reverse_feature = ReverseLayerF.apply(conv_layers_out, alpha)

			output_domain = self.domain_predictor(reverse_feature)

			return normalized_array, output_domain
		
		return normalized_array

backbone = 'resnet34'
#moco_dim = 768
moco_dim = 128
moco_m = 0.999
temperature = 0.07

batch_size = BATCH_SIZE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder = Encoder(dim=moco_dim).to(device)
momentum_encoder = Encoder(dim=moco_dim).to(device)

encoder.embedding.weight.data.normal_(mean=0.0, std=0.01)
encoder.embedding.bias.data.zero_()

#momentum_encoder.embedding.weight.data.normal_(mean=0.0, std=0.01)
#momentum_encoder.embedding.bias.data.zero_()

import torchvision

momentum_encoder.load_state_dict(encoder.state_dict(), strict=False)

for param in momentum_encoder.parameters():
	param.requires_grad = False

#del pre_trained_network

#DATA AUGMENTATION
from torchvision import transforms
prob = 0.75
pipeline_transform_paper = A.Compose([
	#A.RandomScale(scale_limit=(-0.005,0.005), interpolation=2, p=prob),
	#A.RandomCrop(height=220, width=220, p=prob),
	#A.Resize(224,224,always_apply=True),
	#A.MotionBlur(blur_limit=3, p=prob),
	#A.MedianBlur(blur_limit=3, p=prob),
	#A.CropAndPad(percent=(-0.01, -0.05),pad_mode=1,always_apply=True),
	A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1), always_apply=True),
	A.VerticalFlip(p=prob),
	A.HorizontalFlip(p=prob),
	A.RandomRotate90(p=prob),
	#A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),always_apply=True),
	A.ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, always_apply=True),
	A.GaussianBlur (blur_limit=(1, 3), sigma_limit=0, always_apply=True),
	#A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),always_apply=True),
	#A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=prob),
	#A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=prob),
	#A.RandomBrightness(limit=0.2, p=prob),
	#A.RandomContrast(limit=0.2, p=prob),
	#A.GaussNoise(p=prob),
	#A.ElasticTransform(alpha=2,border_mode=4, sigma=20, alpha_affine=20, p=prob, always_apply=True),
	#A.GridDistortion(num_steps=2, distort_limit=0.2, interpolation=1, border_mode=4, p=prob),
	#A.GlassBlur(sigma=0.3, max_delta=2, iterations=1, p=prob),
	#A.OpticalDistortion (distort_limit=0.2, shift_limit=0.2, interpolation=1, border_mode=4, value=None, p=prob),
	#A.GridDropout (ratio=0.3, unit_size_min=3, unit_size_max=40, holes_number_x=3, holes_number_y=3, shift_x=1, shift_y=10, random_offset=True, fill_value=0, p=prob),
	#A.Equalize(p=prob),
	#A.Posterize(p=prob, always_apply=True),
	#A.RandomGamma(p=prob, always_apply=True),
	#A.Superpixels(p_replace=0.05, n_segments=100, max_size=128, interpolation=1, p=prob),
	#A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=prob),
	A.ToGray(p=0.2),
	#A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=0, p=prob),
	#A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=255, p=prob),
	])

pipeline_transform = A.Compose([
	#A.RandomScale(scale_limit=(-0.005,0.005), interpolation=2, p=prob),
	#A.RandomCrop(height=220, width=220, p=prob),
	#A.Resize(224,224,always_apply=True),
	#A.MotionBlur(blur_limit=3, p=prob),
	#A.MedianBlur(blur_limit=3, p=prob),
	#A.CropAndPad(percent=(-0.01, -0.05),pad_mode=1,always_apply=True),
	A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1), p = prob),
	A.VerticalFlip(p=prob),
	A.HorizontalFlip(p=prob),
	A.RandomRotate90(p=prob),
	#A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),always_apply=True),
	A.ColorJitter (brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, always_apply=True),
	A.GaussianBlur (blur_limit=(1, 3), sigma_limit=0, always_apply=True),
	#A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),always_apply=True),
	#A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=prob),
	#A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=prob),
	#A.RandomBrightness(limit=0.2, p=prob),
	#A.RandomContrast(limit=0.2, p=prob),
	#A.GaussNoise(p=prob),
	A.ElasticTransform(alpha=2,border_mode=4, sigma=10, alpha_affine=10, p=prob),
	A.GridDistortion(num_steps=1, distort_limit=0.1, interpolation=1, border_mode=4, p=prob),
	#A.GlassBlur(sigma=0.3, max_delta=2, iterations=1, p=prob),
	A.OpticalDistortion (distort_limit=0.2, shift_limit=0.2, interpolation=1, border_mode=4, value=None, p=prob),
	#A.GridDropout (ratio=0.3, unit_size_min=3, unit_size_max=40, holes_number_x=3, holes_number_y=3, shift_x=1, shift_y=10, random_offset=True, fill_value=0, p=prob),
	#A.Equalize(p=prob),
	#A.Posterize(p=prob, always_apply=True),
	#A.RandomGamma(p=prob, always_apply=True),
	#A.Superpixels(p_replace=0.05, n_segments=100, max_size=128, interpolation=1, p=prob),
	#A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3, p=prob),
	A.ToGray(p=0.2),
	#A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=0, p=prob),
	#A.CoarseDropout (max_holes=20, max_height=10, max_width=10, min_holes=None, min_height=1, min_width=1, fill_value=255, p=prob),
	])

p_soft = 0.5
pipeline_transform_soft = A.Compose([
	#A.ElasticTransform(alpha=0.01,p=p_soft),
	#A.RGBShift (r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=True, p=p_soft),
	A.HueSaturationValue(hue_shift_limit=(-15,8),sat_shift_limit=(-20,10),val_shift_limit=(-8,8),p=p_soft),
	A.VerticalFlip(p=p_soft),
	A.HorizontalFlip(p=p_soft),
	A.RandomRotate90(p=p_soft),
	#A.HueSaturationValue(hue_shift_limit=(-25,10),sat_shift_limit=(-25,15),val_shift_limit=(-15,15),p=p_soft),
	#A.CLAHE(clip_limit=1.0, tile_grid_size=(8, 8), p=p_soft),
	#A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=p_soft),
	#A.RandomBrightness(limit=0.1, p=p_soft),
	#A.RandomContrast(limit=0.1, p=p_soft),
	])
	
#DATA NORMALIZATION
preprocess = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def generate_list_instances(filename):

	instance_dir = WSI_FOLDER
	fname = os.path.split(filename)[-1]
	
	instance_csv = instance_dir+fname+'/'+fname+'_paths_densely.csv'
	
	return instance_csv 


class ImbalancedDatasetSampler_multilabel(torch.utils.data.sampler.Sampler):

	def __init__(self, dataset, indices=None, num_samples=None):

		self.indices = list(range(len(dataset)))             if indices is None else indices

		self.num_samples = len(self.indices)             if num_samples is None else num_samples

		
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			for l in label:
				if l in label_to_count:
					label_to_count[l] += 1
				else:
					label_to_count[l] = 1

		weights = []

		for idx in self.indices:
			c = 0
			for j, l in enumerate(self._get_label(dataset, idx)):
				c = c+(1/label_to_count[l])
				
			weights.append(c/(j+1))
		self.weights = torch.DoubleTensor(weights)
		
	def _get_label(self, dataset, idx):
		labels = np.where(dataset[idx,1:]==1)[0]
		#print(labels)
		#labels = dataset[idx,2]
		return labels

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

class Balanced_Multimodal(torch.utils.data.sampler.Sampler):

	def __init__(self, dataset, indices=None, num_samples=None, alpha = 0.5):

		self.indices = list(range(len(dataset)))             if indices is None else indices

		self.num_samples = len(self.indices)             if num_samples is None else num_samples

		class_sample_count = [0,0,0,0,0]


		class_sample_count = np.sum(train_dataset[:,1:],axis=0)

		min_class = np.argmin(class_sample_count)
		class_sample_count = np.array(class_sample_count)
		weights = []
		for c in class_sample_count:
			weights.append((c/class_sample_count[min_class]))

		ratio = np.array(weights).astype(np.float)

		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			for l in label:
				if l in label_to_count:
					label_to_count[l] += 1
				else:
					label_to_count[l] = 1

		weights = []

		for idx in self.indices:
			c = 0
			for j, l in enumerate(self._get_label(dataset, idx)):
				c = c+(1/label_to_count[l])#*ratio[l]

			weights.append(c/(j+1))
			#weights.append(c)
			
		self.weights_original = torch.DoubleTensor(weights)

		self.weights_uniform = np.repeat(1/self.num_samples, self.num_samples)

		#print(self.weights_a, self.weights_b)

		beta = 1 - alpha
		self.weights = (alpha * self.weights_original) + (beta * self.weights_uniform)


	def _get_label(self, dataset, idx):
		labels = np.where(dataset[idx,1:]==1)[0]
		#print(labels)
		#labels = dataset[idx,2]
		return labels

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(
			self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples

def H_E_Staining(img, Io=240, alpha=1, beta=0.15):
	''' 
	Normalize staining appearence of H&E stained images
	
	Example use:
		see test.py
		
	Input:
		I: RGB input image
		Io: (optional) transmitted light intensity
		
	Output:
		Inorm: normalized image
		H: hematoxylin image
		E: eosin image
	
	Reference: 
		A method for normalizing histology slides for quantitative analysis. M.
		Macenko et al., ISBI 2009
	'''

	# define height and width of image
	h, w, c = img.shape
	
	# reshape image
	img = img.reshape((-1,3))

	# calculate optical density
	OD = -np.log((img.astype(np.float)+1)/Io)
	
	# remove transparent pixels
	ODhat = OD[~np.any(OD<beta, axis=1)]
		
	# compute eigenvectors
	_, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
	
	#eigvecs *= -1
	
	#project on the plane spanned by the eigenvectors corresponding to the two 
	# largest eigenvalues    
	That = ODhat.dot(eigvecs[:,1:3])
	
	phi = np.arctan2(That[:,1],That[:,0])
	
	minPhi = np.percentile(phi, alpha)
	maxPhi = np.percentile(phi, 100-alpha)
	
	vMin = eigvecs[:,1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
	vMax = eigvecs[:,1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
	
	# a heuristic to make the vector corresponding to hematoxylin first and the 
	# one corresponding to eosin second
	if vMin[0] > vMax[0]:
		HE = np.array((vMin[:,0], vMax[:,0])).T
	else:
		HE = np.array((vMax[:,0], vMin[:,0])).T
	
	return HE

class Dataset_instance(data.Dataset):

	def __init__(self, list_IDs, mode):
		self.list_IDs = list_IDs
		#self.list_IDs = list_IDs[:,0]
		#self.list_hes = list_IDs[:,1:]
		self.mode = mode

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		ID = self.list_IDs[index]
		# Load data and get label
		img = Image.open(ID)
		X = np.asarray(img)
		img.close()

		#k = pipeline_transform_soft(image=k)['image']
		#k = pipeline_transform(image=q)['image']

		h_e_matrix = [0,0,0,0,0,0]
		h_e_matrix = np.array(h_e_matrix)

		if (self.mode == 'train'):

			#h_e_matrix = self.list_hes[index].tolist()

			b = False
			k = X				

			while(b==False):

				k = pipeline_transform_soft(image=k)['image']

				try:
					h_e_matrix = H_E_Staining(k)
					b = True
				except:
					k = pipeline_transform_soft(image=k)['image']
					pass
					
					#idx_n = np.random.randint(0,self.__len__())
					#self.__getitem__(idx_n)
					#pass

			q = pipeline_transform_paper(image=k)['image']
			h_e_matrix = np.reshape(h_e_matrix, 6)
			h_e_matrix = np.asarray(h_e_matrix)
			
			
			#h_e_matrix = np.reshape(h_e_matrix, 6)
			#h_e_matrix = np.array(h_e_matrix)

			#print(h_e_matrix)
		#print(q.shape)
		else:
			k = X
			q = pipeline_transform(image=k)['image']

		del X
		#data transformation
		q = preprocess(q).type(torch.FloatTensor)
		k = preprocess(k).type(torch.FloatTensor)
		h_e_matrix = torch.FloatTensor(h_e_matrix)
		#return input_tensor
		return k, q, h_e_matrix

class Dataset_bag(data.Dataset):

	def __init__(self, list_IDs, labels):

		self.labels = labels
		self.list_IDs = list_IDs

	def __len__(self):

		return len(self.list_IDs)

	def __getitem__(self, index):
		# Select sample
		WSI = self.list_IDs[index]

		return WSI

#parameters bag
batch_size_bag = 16

"""
sampler = ImbalancedDatasetSampler_multilabel
params_train_bag = {'batch_size': batch_size_bag,
		'sampler': sampler(train_dataset)}
		#'shuffle': True}
"""
#"""
sampler = Balanced_Multimodal
params_train_bag = {'batch_size': batch_size_bag,
		#'sampler': sampler(train_dataset,alpha=0.25)}
		'shuffle': True}
#"""
"""
sampler = Balanced_Multimodal
params_bag_train = {'batch_size': batch_size_bag,
		'sampler': sampler(train_dataset,alpha=0.5)}
		#'shuffle': True}
"""


params_bag_test = {'batch_size': batch_size_bag,
		#'sampler': sampler(train_dataset)
	  'shuffle': True}

params_bag_train_queue = {'batch_size': int(batch_size_bag*2),
		'sampler': sampler(train_dataset,alpha=0.25)}
	  #'shuffle': True}

params_bag_test_queue = {'batch_size': int(batch_size_bag*2),
		#'sampler': sampler(train_dataset)
	  'shuffle': True}

training_set_bag = Dataset_bag(train_dataset[:,0], train_dataset[:,1:])
training_generator_bag = data.DataLoader(training_set_bag, **params_train_bag)

#validation_set_bag = Dataset_bag(valid_dataset[:,0], valid_dataset[:,1:])
#validation_generator_bag = data.DataLoader(validation_set_bag, **params_bag_test)

training_set_bag = Dataset_bag(train_dataset[:,0], train_dataset[:,1:])
training_generator_bag_queue = data.DataLoader(training_set_bag, **params_bag_train_queue)

#validation_set_bag = Dataset_bag(valid_dataset[:,0], valid_dataset[:,1:])
#validation_generator_bag_queue = data.DataLoader(validation_set_bag, **params_bag_test_queue)

#params patches generated

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in encoder.parameters())
print(f'{total_params:,} total parameters.')

total_trainable_params = sum(
	p.numel() for p in encoder.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

torch.backends.cudnn.benchmark=True

class RMSELoss(torch.nn.Module):
	def __init__(self, eps=1e-6):
		super().__init__()
		self.mse = torch.nn.MSELoss()
		self.eps = eps
		
	def forward(self,yhat,y):
		loss = torch.sqrt(self.mse(yhat,y) + self.eps)
		return loss

def loss_function(q, k, queue):

	#N is the batch size
	N = q.shape[0]
	
	#C is the dimension of the representation
	C = q.shape[1]

	#BMM stands for batch matrix multiplication
	#If mat1 is B × n × M tensor, then mat2 is B × m × P tensor,
	#Then output a B × n × P tensor.
	pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),temperature))
	
	#Matrix multiplication is performed between the query and the queue tensor
	neg = torch.sum(torch.exp(torch.div(torch.mm(q.view(N,C), torch.t(queue)),temperature)), dim=1)
   
	#Sum up
	denominator = neg + pos

	return torch.mean(-torch.log(torch.div(pos,denominator)))

criterion = torch.nn.CrossEntropyLoss().to(device)
#criterion_domain = RMSELoss().to(device)
criterion_domain = torch.nn.L1Loss()

lambda_val = 0.5

import torch.optim as optim
optimizer_str = 'adam'
#optimizer_str = 'sgd'
#optimizer_str = 'lars'

#print(model.conv_layers.parameters())

# Optimizer
SGD_momentum = 0.9
weight_decay = 1e-4
shuffle_bn = True

if (optimizer_str=='sgd'):
	optimizer = optim.SGD(encoder.parameters(), 
						lr=lr, 
						momentum=SGD_momentum, 
						weight_decay=weight_decay)

elif(optimizer_str=='lars'):
	optimizer = LARS(params=encoder.parameters(),
						lr=lr,
						momentum=SGD_momentum,
						weight_decay=weight_decay,
						eta=0.001,
						max_epoch=EPOCHS)

elif (optimizer_str=='adam'):
	optimizer = optim.Adam(encoder.parameters(),lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=True)			
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def momentum_step(m=1):
	'''
	Momentum step (Eq (2)).
	Args:
		- m (float): momentum value. 1) m = 0 -> copy parameter of encoder to key encoder
									 2) m = 0.999 -> momentum update of key encoder
	'''
	params_q = encoder.state_dict()
	params_k = momentum_encoder.state_dict()
	
	dict_params_k = dict(params_k)
	
	for name in params_q:
		theta_k = dict_params_k[name]
		theta_q = params_q[name].data
		dict_params_k[name].data.copy_(m * theta_k + (1-m) * theta_q)

	momentum_encoder.load_state_dict(dict_params_k)

def update_lr(epoch):
	'''
	Learning rate scheduling.
	Args:
		- epoch (float): Set new learning rate by a given epoch.
	'''
	
	if epoch < 10:
		lr = args.lr
	elif epoch >= 10 and epoch < 20:
		lr = args.lr * 0.1
	elif epoch >= 20 :
		lr = args.lr * 0.01
	
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def update_queue(queue, k):

	len_k = k.shape[0]
	len_queue = queue.shape[0]

	new_queue = torch.cat([k, queue], dim=0)

	new_queue = new_queue[:num_keys]

	return new_queue
	
''' ######################## < Step 4 > Start training ######################## '''

# Initialize momentum_encoder with parameters of encoder.
momentum_step(m=0)

def mapping_patches(patches, THRESHOLD):
	
	n_patches = len(patches)
	ratio = int(n_patches/THRESHOLD)
	#print(ratio)
	
	if (ratio>0):
		
		#idx_list = np.random.randint(0, n_patches, THRESHOLD)        
		#idx = list(set(idx_list))
		#new_patches = patches[idx]
		
		idx_list = np.random.choice(n_patches, THRESHOLD, replace = False)
		idx = list(set(idx_list))
		idx.sort()
		new_patches = patches[idx]
		
	else:
		
		new_patches = patches
	
	return new_patches

def validate(epoch, generator):
	#accumulator for validation set

	encoder.eval()
	momentum_encoder.eval()

	queue = []
	dataloader_iterator = iter(validation_generator_bag_queue)

	wsis = next(dataloader_iterator)
			
	fnames_patches = []

	new_patches = 0

	for wsi in wsis:

		fname = wsi

		print(fname)

		csv_fname = generate_list_instances(fname)
		csv_instances = pd.read_csv(csv_fname, sep=',', header=None).values

		l_csv = len(csv_instances)
		#csv_instances = mapping_patches(csv_instances,img_len)
		p_to_select = 512
		csv_instances = mapping_patches(csv_instances,p_to_select)

		new_patches = new_patches + len(csv_instances)

		fnames_patches = np.append(fnames_patches, csv_instances)
	
	#fnames_patches = np.reshape(fnames_patches, (new_patches, 7))
	#print(fnames_patches.shape)

	num_workers = 2
	params_instance = {'batch_size': batch_size,
			'shuffle': True,
			'num_workers': num_workers}

	instances = Dataset_instance(fnames_patches, 'valid')
	generator_inst = data.DataLoader(instances, **params_instance)

	with torch.no_grad():
		for i, (_, img, _) in enumerate(generator_inst):
			key_feature = momentum_encoder(img.to(device), 'valid', _)
			queue.append(key_feature)

			if i == (num_keys / batch_size) - 1:
				break
		queue = torch.cat(queue, dim=0)

	valid_loss = 0.0
	total_iters = 0
	dataloader_iterator = iter(generator)

	j = 0

	iterations = int(len(valid_dataset) / batch_size_bag)

	for i in range(iterations):
		print('[%d], %d / %d ' % (epoch, i, iterations))

		try:
			wsis = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(training_generator_bag)
			wsis = next(dataloader_iterator)
			#inputs: bags, labels: labels of the bags

		fnames_patches = []

		new_patches = 0

		for wsi in wsis:

			fname = wsi

			print(fname)

			csv_fname = generate_list_instances(fname)
			csv_instances = pd.read_csv(csv_fname, sep=',', header=None).values

			l_csv = len(csv_instances)
			#csv_instances = mapping_patches(csv_instances,img_len)
			p_to_select = max(int(l_csv/3), img_len)
			csv_instances = mapping_patches(csv_instances,p_to_select)

			new_patches = new_patches + len(csv_instances)

			fnames_patches = np.append(fnames_patches, csv_instances)
	
		#fnames_patches = np.reshape(fnames_patches, (new_patches, 7))

		num_workers = 2
		params_instance = {'batch_size': batch_size,
				'shuffle': True,
				'num_workers': num_workers}

		instances = Dataset_instance(fnames_patches, mode)
		generator = data.DataLoader(instances, **params_instance)

		for a, (x_q, x_k, _) in enumerate(generator):
			# Preprocess
			
			momentum_encoder.zero_grad()
			encoder.zero_grad()

			# Shffled BN : shuffle x_k before distributing it among GPUs (Section. 3.3)
			if shuffle_bn:
				idx = torch.randperm(x_k.size(0))
				x_k = x_k[idx]
				
			# x_q, x_k : (N, 3, 64, 64)            
			x_q, x_k = x_q.to(device), x_k.to(device)

			q = encoder(x_q, 'valid', _) # q : (N, 128)
			k = momentum_encoder(x_k, 'valid', _).detach()
			
			# Shuffled BN : unshuffle k (Section. 3.3)
			if shuffle_bn:
				k_temp = torch.zeros_like(k)
				for a, j in enumerate(idx):
					k_temp[j] = k[a]
				k = k_temp

			"""
			# positive logits: Nx1
			l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
			# negative logits: NxK
			l_neg = torch.einsum('nc,ck->nk', [q, queue.t()])

			# Positive sampling q & k
			#l_pos = torch.sum(q * k, dim=1, keepdim=True) # (N, 1)
			#print("l_pos",l_pos)

			# Negative sampling q & queue
			#l_neg = torch.mm(q, queue.t()) # (N, 4096)
			#print("l_neg",l_neg)

			# Logit and label
			logits = torch.cat([l_pos, l_neg], dim=1) / temperature # (N, 4097) witi label [0, 0, ..., 0]
			labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)

			# Get loss and backprop
			loss_moco = criterion(logits, labels)
			"""
			loss = loss_function(q, k, queue)

			# Encoder update
			#optimizer.step()

			# Momentum encoder update
			#momentum_step(m=moco_m)

			# Update dictionary
			#queue = torch.cat([k, queue[:queue.size(0) - k.size(0)]], dim=0)
			queue = update_queue(queue, k)
			#print(queue.shape)

			# Print a training status, save a loss value, and plot a loss graph.
			
			valid_loss = valid_loss + ((1 / (total_iters+1)) * (loss.item() - valid_loss)) 
			total_iters = total_iters + 1
			print('[Epoch : %d / Total iters : %d] : loss : %f ...' %(epoch, total_iters, valid_loss))

			momentum_encoder.zero_grad()
			encoder.zero_grad()

			torch.cuda.empty_cache()
	return valid_loss


# Training
print('\nStart training!')
epoch = 0 

iterations_per_epoch = 8600

losses_train = []

	#number of epochs without improvement
EARLY_STOP_NUM = 10
early_stop_cont = 0
epoch = 0
num_epochs = EPOCHS
validation_checkpoints = checkpoint_path+'validation_losses/'
create_dir(validation_checkpoints)
	#number of epochs without improvement
epoch = 0
iterations = int(len(train_dataset) / batch_size_bag)#+100
#iterations = 600

tot_batches_training = iterations#int(len(train_dataset)/batch_size_bag)
best_loss = 100000.0

tot_iterations = num_epochs * iterations_per_epoch
cont_iterations_tot = 0

TEMPERATURE = 0.07

p_to_select = 512

while (epoch<num_epochs and early_stop_cont<EARLY_STOP_NUM):
	total_iters = 0 
	
	#accumulator loss for the outputs
	train_loss = 0.0
	train_loss_domain = 0.0
	train_loss_moco = 0.0
	
	#if loss function lower
	is_best = False

	print('\n[3 / 3]. Initializing a queue with %d keys.' % num_keys)
	queue = []
	dataloader_iterator = iter(training_generator_bag_queue)

	wsis = next(dataloader_iterator)
			
	fnames_patches = []

	new_patches = 0

	

	for wsi in wsis:

		fname = wsi

		print(fname)

		csv_fname = generate_list_instances(fname)
		csv_instances = pd.read_csv(csv_fname, sep=',', header=None).values

		l_csv = len(csv_instances)
		#csv_instances = mapping_patches(csv_instances,img_len)
		
		csv_instances = mapping_patches(csv_instances,p_to_select)

		new_patches = new_patches + len(csv_instances)

		fnames_patches = np.append(fnames_patches, csv_instances)
	
	#fnames_patches = np.reshape(fnames_patches, (new_patches, 7))
	#print(fnames_patches.shape)

	num_workers = 2
	params_instance = {'batch_size': batch_size,
			'shuffle': True,
			'num_workers': num_workers}

	instances = Dataset_instance(fnames_patches, 'valid')
	generator = data.DataLoader(instances, **params_instance)

	with torch.no_grad():
		for i, (_, img, _) in enumerate(generator):
			key_feature = momentum_encoder(img.to(device), 'valid', _)
			queue.append(key_feature)

			if i == (num_keys / batch_size) - 1:
				break
		queue = torch.cat(queue, dim=0)

	
	dataloader_iterator = iter(training_generator_bag)

	j = 0

	mode = 'train'

	encoder.train()
	momentum_encoder.train()

	for i in range(iterations):
		print('[%d], %d / %d ' % (epoch, i, tot_batches_training))

		try:
			wsis = next(dataloader_iterator)
		except StopIteration:
			dataloader_iterator = iter(training_generator_bag)
			wsis = next(dataloader_iterator)
			#inputs: bags, labels: labels of the bags

		fnames_patches = []

		new_patches = 0

		for wsi in wsis:

			fname = wsi

			print(fname)

			csv_fname = generate_list_instances(fname)
			csv_instances = pd.read_csv(csv_fname, sep=',', header=None).values
			l_csv = len(csv_instances)
			p_to_select = max(int(l_csv/3), p_to_select)
			
			csv_instances = mapping_patches(csv_instances,p_to_select)
			new_patches = new_patches + len(csv_instances)

			fnames_patches = np.append(fnames_patches, csv_instances)
		
		#fnames_patches = np.reshape(fnames_patches, (new_patches, 7))

		num_workers = 2
		params_instance = {'batch_size': batch_size,
				'shuffle': True,
				'num_workers': num_workers}

		instances = Dataset_instance(fnames_patches, mode)
		generator = data.DataLoader(instances, **params_instance)

		#print("data ready")

		for a, (x_q, x_k, he_staining) in enumerate(generator):
			
			p = float(cont_iterations_tot + epoch * tot_iterations) / num_epochs / tot_iterations

			alpha = 2. / (1. + np.exp(-10 * p)) - 1

			# Preprocess
			#momentum_encoder.train()
			#momentum_encoder.zero_grad()
			#encoder.train()
			#encoder.zero_grad()

			# Shffled BN : shuffle x_k before distributing it among GPUs (Section. 3.3)
			if shuffle_bn:
				idx = torch.randperm(x_k.size(0))
				x_k = x_k[idx]
				
			# x_q, x_k : (N, 3, 64, 64)            
			x_q, x_k, he_staining = x_q.to(device), x_k.to(device), he_staining.type(torch.FloatTensor).to(device)

			q, he_q = encoder(x_q, mode, alpha) # q : (N, 128)
			with torch.no_grad():
				k = momentum_encoder(x_k, 'valid', _).detach() # k : (N, 128)
			
			# Shuffled BN : unshuffle k (Section. 3.3)
			if shuffle_bn:
				k_temp = torch.zeros_like(k)
				for a, j in enumerate(idx):
					k_temp[j] = k[a]
				k = k_temp
			"""
			# positive logits: Nx1
			l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
			# negative logits: NxK
			l_neg = torch.einsum('nc,ck->nk', [q, queue.t()])

			# Positive sampling q & k
			#l_pos = torch.sum(q * k, dim=1, keepdim=True) # (N, 1)
			#print("l_pos",l_pos)

			# Negative sampling q & queue
			#l_neg = torch.mm(q, queue.t()) # (N, 4096)
			#print("l_neg",l_neg)

			# Logit and label
			logits = torch.cat([l_pos, l_neg], dim=1) / temperature # (N, 4097) witi label [0, 0, ..., 0]
			labels = torch.zeros(logits.size(0), dtype=torch.long).to(device)

			# Get loss and backprop
			loss_moco = criterion(logits, labels)
			"""
			loss_moco = loss_function(q, k, queue)
			loss_domains = lambda_val * criterion_domain(he_q, he_staining)

			loss = loss_moco + loss_domains

			loss.backward()

			# Encoder update
			optimizer.step()

			momentum_encoder.zero_grad()
			encoder.zero_grad()

			# Momentum encoder update
			momentum_step(m=moco_m)

			# Update dictionary
			#queue = torch.cat([k, queue[:queue.size(0) - k.size(0)]], dim=0)
			queue = update_queue(queue, k)
			#print(queue.shape)
			
			# Print a training status, save a loss value, and plot a loss graph.
			
			train_loss_moco = train_loss_moco + ((1 / (total_iters+1)) * (loss_moco.item() - train_loss_moco)) 
			train_loss_domain = train_loss_domain + ((1 / (total_iters+1)) * (loss_domains.item() - train_loss_domain)) 
			total_iters = total_iters + 1
			cont_iterations_tot = cont_iterations_tot + 1
			train_loss = train_loss_moco + train_loss_domain

			print('[Epoch : %d / Total iters : %d] : loss_moco :%f, loss_domain :%f ...' %(epoch, total_iters, train_loss_moco, train_loss_domain))
			
		if (i%10==True):
			print('a')
			if (best_loss>train_loss_moco):
				early_stop_cont = 0
				print ("=> Saving a new best model")
				print("previous loss : " + str(best_loss) + ", new loss function: " + str(train_loss_moco))
				best_loss = train_loss_moco
				try:
					torch.save(encoder.state_dict(), model_weights_filename,_use_new_zipfile_serialization=False)
				except:
					torch.save(encoder.state_dict(), model_weights_filename)
			else:

				try:
					torch.save(encoder.state_dict(), model_weights_temporary_filename,_use_new_zipfile_serialization=False)
				except:
					torch.save(encoder.state_dict(), model_weights_temporary_filename)

			torch.cuda.empty_cache()
		
		# Update learning rate
	#update_lr(epoch)

	print("epoch "+str(epoch)+ " train loss: " + str(train_loss))
	
	print("evaluating validation")
	"""
	valid_loss = validate(epoch, validation_generator_bag)

	#save validation
	filename_val = validation_checkpoints+'validation_value_'+str(epoch)+'.csv'
	array_val = [valid_loss]
	File = {'val':array_val}
	df = pd.DataFrame(File,columns=['val'])
	np.savetxt(filename_val, df.values, fmt='%s',delimiter=',')

	#save_hyperparameters
	filename_hyperparameters = checkpoint_path+'hyperparameters.csv'
	array_lr = [str(lr)]
	array_opt = [optimizer_str]
	array_wt_decay = [str(weight_decay)]
	array_embedding = [EMBEDDING_bool]
	File = {'opt':array_opt, 'lr':array_lr,'wt_decay':array_wt_decay,'array_embedding':EMBEDDING_bool}

	df = pd.DataFrame(File,columns=['opt','lr','wt_decay','array_embedding'])
	np.savetxt(filename_hyperparameters, df.values, fmt='%s',delimiter=',')
	"""


	

	
	epoch = epoch+1
	if (early_stop_cont == EARLY_STOP_NUM):
		print("EARLY STOPPING")