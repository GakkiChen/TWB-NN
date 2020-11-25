#Trimmable Wide Binary Neural Network

## Implementation for CIFAR-10 classification.


train.py:     train the binary model 

my_model.py:    create the TWB-Net network

new_layers.py:      define quantization functions

utils.py:  define auxiliary functions

## How to run:

## Train binary ResNet-20 with M=5:
python train.py

## Prune ResNet-20 and fine-tune:
python train.py --trimming

## Train with the pruned network architecture from scratch (TWB-Net as a NAS method):
python train.py --ResNet20_prunedArch
