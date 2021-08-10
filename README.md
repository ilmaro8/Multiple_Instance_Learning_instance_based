# Multiple_Instance_Learning
Implementation of Multiple Instance Learning instance-based CNNs for histopathology images classification.

## Reference
If you find this repository useful in your research, please cite: ...

Paper link: 

## Requirements
Python==3.6.9, albumentations==0.1.8, numpy==1.17.3, opencv==4.2.0, pandas==0.25.2, pillow==6.1.0, torchvision==0.3.0, pytorch==1.1.0

## Best models
- The best models for the Multiple Instance Learning CNN is available [here](https://drive.google.com/drive/folders/1jLWLRzIYTphFW-ywyODa7yozyWtQviJP).

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

## Pre-Processing
The WSIs are split in 224x224 pixels patches, from magnification 10x. 
The methods used to extract the patches come from [Multi_Scale_Tools library](https://github.com/sara-nl/multi-scale-tools)

## CSV Input Files:
CSV files are used as input for the scripts. The csvs have the following structures
- For each partition (train, validation, test), the csv file has id_img, cancer, high-grade dysplasia, low-grade dysplasia, hyperplastic polyp, normal glands as column.
- For the patches used as test set, the csv file has path_path, label as column.

## Training
Script to train the CNN at WSI-level, using an instance-based MIL CNN:
python train.py -c resnet34 -b 512 -p att -e 10 -t multilabel -f True -i /PATH/WHERE/TO/FIND/THE/CSVS/INCLUDING/THE/PARTITIONS -o /PATH/WHERE/TO/SAVE/THE/MODEL/WEIGHTS -w /PATH/WHERE/TO/FIND/THE/PATCHES

## Testing
### WSI-level
Script to test the CNN at WSI-level.
python testing_WSI.py -c resnet34 -b 512 -p att -t multilabel -f True -m /PATH/TO/MODEL/WEIGHTS.pt -i /PATH/TO/INPUT/CSV.csv -w /PATH/WHERE/TO/FIND/THE/PATCHES

### patch-level 
python testing_patches.py -c resnet34 -b 512 -p att -t multilabel -f True -m /PATH/TO/MODEL/WEIGHTS.pt -i /PATH/TO/INPUT/CSV.csv

## Acknoledgements
This project has received funding from the EuropeanUnion’s Horizon 2020 research and innovation programme under grant agree-ment No. 825292 [ExaMode](http://www.examode.eu). Infrastructure fromthe SURFsara HPC center was used to train the CNN models in parallel. Otálora thanks Minciencias through the call 756 for PhD studies.
