# Unified Perceptual Parsing for Scene Understanding (Under Construction)

This is a pyTorch implementation of the unified perceptual parsing network on Broden+ dataset and ADE20K dataset. This work is published at ECCV'18 [Unified Perceptual Parsing for Scene Understanding]().

Broden+ dataset is the standardized Broden dataset, previously proposed in [Network Dissection](https://github.com/CSAILVision/NetDissect). ADE20K dataset is a recent image dataset for [scene parsing](https://github.com/CSAILVision/semantic-segmentation-pytorch). 

## Use pretrained models

You can use our pretrained models in PyTorch to segment input image. The usage is as follows:

```

```

## Train and evaluate on Broden+

You can train the networks and evaluate them on the Broden+ dataset

1. Download the Broden+ dataset.
```

```

2. Run the training script.
```

```

3. Evaluate the trained model.
```

```

## Reference

If you find the code or the pretrained models useful, please consider to cite the following papers:

Unified Perceptual Parsing for Scene Understanding. T. Xiao, Y. Liu, B. Zhou, Y. Jiang, and J. Sun. arXiv preprint
```
    @article{xiao2018unified,
      title={Unified Perceptual Parsing for Scene Understanding},
      author={Xiao, Tete and Liu, Yingcheng and Zhou, Bolei and Jiang, Yuning and Sun, Jian},
      journal={European Conference on Computer Vision},
      year={2018}
    }
``` 
