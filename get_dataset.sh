
# get pascal voc
# standard voc
mkdir -p data
cd data
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
wget -c http://pjreddie.com/media/files/VOC2012test.tar
tar -xvf VOC2012test.tar

# voc aug
wget -c wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar -zxvf benchmark.tgz
mkdir -p VOC_AUG VOC_AUG/images VOC_AUG/labels
cp -riv benchmark_RELEASE/dataset/*txt VOC_AUG/
cp -riv benchmark_RELEASE/dataset/img/* VOC_AUG/images/
cd ..
python3 utils/convert_pascal_aug.py

# get pretrained init model
# mkdir -p pretrained
# cd pretrained
# wget -c http://www.cs.jhu.edu/~alanlab/ccvl/init_models/vgg16_20M.caffemodel
# wget -c http://liangchiehchen.com/projects/released/deeplab_coco_largefov/vgg16_20M_coco.caffemodel.zip
# unzip vgg16_20M_coco.caffemodel.zip
# cd ..
