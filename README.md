# Multiple_Instance_Learning
Implementation of Multiple Instance Learning instance-based CNNs for histopathology images classification.

## Reference
If you find this repository useful in your research, please cite: N.Marini et al. (2022). "Unleashing the potential of digital pathology data by training computer-aided diagnosis models without human annotations"

Paper link: https://www.nature.com/articles/s41746-022-00635-4

## Requirements
Python==3.6.9, albumentations==0.1.8, numpy==1.17.3, opencv==4.2.0, pandas==0.25.2, pillow==6.1.0, torchvision==0.8.1, pytorch==1.7.0

## Best models
- The best models for the Multiple Instance Learning CNN is available [here](https://drive.google.com/drive/folders/1-b3YJyJyydxMQPihVGGhQY15lrSH8dJo?usp=sharing).

## Datasets
### Private datasets
Two private datasets are used for training the CNNs:
- AOEC
- Radboudumc
### Publicly available datasets: 
Six publicly available datasets are used for testing the CNNs:
- [GlaS](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/)
- [CRC](https://warwick.ac.uk/fac/cross_fac/tia/data/crc_grading/) 
- [UNITPATHO](https://ieee-dataport.org/open-access/unitopatho) 
- [TCGA-COAD](https://portal.gdc.cancer.gov/projects/TCGA-COAD) 
- [Xu](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-017-1685-x) 
- [AIDA](https://datahub.aida.scilifelab.se/10.23698/aida/drco) 
- [IMP-CRC](https://www.nature.com/articles/s41598-021-93746-z#data-availability).

## Pre-Processing
The WSIs are split in 224x224 pixels patches, from magnification 10x. 
The methods used to extract the patches come from [Multi_Scale_Tools library](https://github.com/sara-nl/multi-scale-tools)

The method is in the /preprocessing folder of the Multi_Scale_Tools library: 
- python Patch_Extractor_Dense_Grid.py -m 10 -w 1.25 -p 10 -r True -s 224 -x 0.7 -y 0 -i /PATH/CSV/IMAGES/TO/EXTRACT.csv -t /PATH/TISSUE/MASKS/TO/USE/ -o /FOLDER/WHERE/TO/STORE/THE/PATCHES/

More info: https://www.frontiersin.org/articles/10.3389/fcomp.2021.684521/full

## CSV Input Files:
CSV files are used as input for the scripts. The csvs have the following structures
- For each partition (train, validation, test), the csv file has id_img, cancer, high-grade dysplasia, low-grade dysplasia, hyperplastic polyp, normal glands as column.
- For the patches used as test set, the csv file has path_path, label as column.

## Training
Script to train the CNN at WSI-level, using an instance-based MIL CNN:
- python train.py -c resnet34 -b 512 -p att -e 10 -t multilabel -f True -i /PATH/WHERE/TO/FIND/THE/CSVS/INCLUDING/THE/PARTITIONS -o /PATH/WHERE/TO/SAVE/THE/MODEL/WEIGHTS -w /PATH/WHERE/TO/FIND/THE/PATCHES

Script to pre-train the CNN using MoCo: python train_MoCo_HE_adversarial_loss.py -c resnet34 -b 512 -p att -e 10 -t multilabel -f True -l 0.001 -i /PATH/WHERE/TO/FIND/THE/CSVS/INCLUDING/THE/PARTITIONS -o /PATH/WHERE/TO/SAVE/THE/MODEL/WEIGHTS -w /PATH/WHERE/TO/FIND/THE/PATCHES

## Testing
### WSI-level
Script to test the CNN at WSI-level.
- python testing_WSI.py -c resnet34 -b 512 -p att -t multilabel -f True -m /PATH/TO/MODEL/WEIGHTS.pt -i /PATH/TO/INPUT/CSV.csv -w /PATH/WHERE/TO/FIND/THE/PATCHES

### patch-level 
- python testing_patches.py -c resnet34 -b 512 -p att -t multilabel -f True -m /PATH/TO/MODEL/WEIGHTS.pt -i /PATH/TO/INPUT/CSV.csv

## Acknoledgements
This project has received funding from the EuropeanUnion’s Horizon 2020 research and innovation programme under grant agree-ment No. 825292 [ExaMode](http://www.examode.eu). Infrastructure fromthe SURFsara HPC center was used to train the CNN models in parallel. Otálora thanks Minciencias through the call 756 for PhD studies.
