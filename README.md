# Unified Perceptual Parsing for Scene Understanding (Under Construction)

This is a pyTorch implementation of Unified Perceptual Parsing network on Broden+ dataset and ADE20K dataset. This work is published at ECCV'18 [Unified Perceptual Parsing for Scene Understanding](), to which [Tete Xiao](http://tetexiao.com), Yingcheng Liu, and [Bolei Zhou](http://people.csail.mit.edu/bzhou/) contribute equally.

Broden+ dataset is the standardized Broden dataset, previously proposed in [Network Dissection](https://github.com/CSAILVision/NetDissect). ADE20K dataset is a recent image dataset for [scene parsing](https://github.com/CSAILVision/semantic-segmentation-pytorch). 

<img src="./teaser/upp_demo.jpg" width="900"/>

## What is Unified Perceptual Parsing?

The human visual system is able to extract a remarkable amount of semantic information from a single glance. We not only instantly parse the objects contained within, but also identify the fine-grained attributes of objects, such as their parts, textures and materials. Motivated by this, we define the task of Unified Perceptual Parsing as the recognition of many visual concepts as possible from a given image. Possible visual concepts are organized into several levels: from scene labels, objects, and parts of objects, to materials and textures of objects.

<img src="./teaser/result_samples.jpg" width="900"/>
[From left to right (inference results): scene classification, and object, part, material, and texture parsing]


## Use pretrained models

### Pertrained models for semantic segmentation on ADE20K

We have released the UPerNet with state-of-the-art performance proposed in our paper as baseline for parsing. UPerNet is based on Feature Pyramid Network (FPN) and Pyramid Pooling Module (PPM), with down-sampling rate of 4, 8 and 16. It doesn't need dilated convolution, a operator that is time-and-memory consuming. Without bells and whistles, it is comparable or even better compared with PSPNet, while requires much shorter training time and less GPU memory. E.g., you cannot train a PSPNet-101 on TITAN Xp GPUs with only 12GB memory, while you can train a UPerNet-101 on such GPUs. Please refer to [semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) for codes and models.

### Pretrained models for unified perceptual parsing on Broden+

We are working to implement UPerNet trained for UPP in PyTorch. Please stay tuned! :)

You can use our pretrained models in PyTorch to segment input image. The usage is as follows:

```

```

## Training on Broden+

You can train the networks and evaluate them on the Broden+ dataset

1. Download and process the Broden+ dataset.
```
    chmod +w download_Broden+.sh
    ./download_Broden+.sh
```
This script downloads and saves the Broden+ dataset to the ```broden_dataset``` directory, then resizes the images whose both sides are largers than 512 pixels.

2. Training a default network (ResNet-50 + UPerNet) for scene classifcation + object, part and material parsing. During training, checkpoints will be saved in folder ```ckpt```.
```
python3 train.py --num_gpus 8 --arch_encoder resnet50 --arch_decoder upernet 
--segm_downsampling_rate 4 --padding_constant 32 --num_epoch 40
```

3. Evaluate the trained model.
```

```

## Reference

If you find the code or the pretrained models useful, please consider to cite the following paper:

```
    @article{xiao2018unified,
      title={Unified Perceptual Parsing for Scene Understanding},
      author={Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
      journal={European Conference on Computer Vision},
      year={2018}
    }
``` 
