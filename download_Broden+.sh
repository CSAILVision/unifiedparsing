#!/usr/bin/env bash
set -e

# PASCAL 2010 Images
if [ ! -f broden_dataset/pascal/VOC2010/ImageSets/Segmentation/train.txt ]
then

echo "Downloading Pascal VOC2010 images"
mkdir -p broden_dataset/pascal
pushd broden_dataset/pascal
wget --progress=bar \
   http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar \
   -O VOCtrainval_03-May-2010.tar
tar xvf VOCtrainval_03-May-2010.tar
rm VOCtrainval_03-May-2010.tar
mv VOCdevkit/* .
rmdir VOCdevkit
popd

fi


# PASCAL Part dataset
if [ ! -f broden_dataset/pascal/part/part2ind.m ]
then

echo "Downloading Pascal Part Dataset"
mkdir -p broden_dataset/pascal/part
pushd broden_dataset/pascal/part
wget --progress=bar \
   http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/trainval.tar.gz \
   -O trainval.tar.gz
tar xvfz trainval.tar.gz
rm trainval.tar.gz
popd

fi


# PASCAL Context dataset
if [ ! -f broden_dataset/pascal/context/labels.txt ]
then

echo "Downloading Pascal Context Dataset"
mkdir -p broden_dataset/pascal/context
pushd broden_dataset/pascal/context
wget --progress=bar \
   http://www.cs.stanford.edu/~roozbeh/pascal-context/trainval.tar.gz \
   -O trainval.tar.gz --no-check-certificate
tar xvfz trainval.tar.gz
rm trainval.tar.gz
popd

fi


# DTD
if [ ! -f broden_dataset/dtd/dtd-r1.0.1/imdb/imdb.mat ]
then

echo "Downloading Describable Textures Dataset"
mkdir -p broden_dataset/dtd
pushd broden_dataset/dtd
wget --progress=bar \
   https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz \
   -O dtd-r1.0.1.tar.gz
tar xvzf dtd-r1.0.1.tar.gz
mv dtd dtd-r1.0.1
rm dtd-r1.0.1.tar.gz
popd

fi


# OpenSurfaces
if [ ! -f broden_dataset/opensurfaces/photos.csv ]
then

echo "Downloading OpenSurfaces Dataset"
mkdir -p broden_dataset/opensurfaces
pushd broden_dataset/opensurfaces
wget --progress=bar \
   http://labelmaterial.s3.amazonaws.com/release/opensurfaces-release-0.zip \
   -O opensurfaces-release-0.zip
unzip opensurfaces-release-0.zip
rm opensurfaces-release-0.zip
PROCESS=process_opensurfaces_release_0.py
popd
cp broden_dataset_utils/$PROCESS broden_dataset/opensurfaces/
pushd broden_dataset/opensurfaces
python3 $PROCESS
popd

fi


# ADE20K
if [ ! -f broden_dataset/ade20k/index_ade20k.mat ]
then

echo "Downloading ADE20K Dataset"
mkdir -p broden_dataset/ade20k
pushd broden_dataset/ade20k
wget --progress=bar \
   http://groups.csail.mit.edu/vision/datasets/ADE20K/ADE20K_2016_07_26.zip \
   -O ADE20K_2016_07_26.zip
unzip ADE20K_2016_07_26.zip
rm ADE20K_2016_07_26.zip
popd

fi

# Resize ADE20 and Opensurfaces
echo "Resize images"
python3 broden_dataset_utils/resize_images.py
echo "Resize done!"
