import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir
from os.path import join

classes = ['bowl']
#classes = ['banana']
txt_folder = '/workspace/Deeplab-Large-FOV/data/VOCdevkit/VOC2012/ImageSets/Segmentation/trainclbs.txt'
txt_folder = '/workspace/Deeplab-Large-FOV/data/VOCdevkit/VOC2012/ImageSets/Segmentation/testclbs.txt'
listfile = '/workspace/Deeplab-Large-FOV/data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
testlistfile = '/workspace/Deeplab-Large-FOV/data/VOCdevkit/VOC2012/ImageSets/Segmentation/test.txt'
list_folder = '/home/kris/github/coco2pascal2txt/jpg_pathList/'
xml_folder = '/workspace/Deeplab-Large-FOV/data/VOCdevkit/VOC2012/Annotations/'
jpg_folder = '<path-to-jpg>'
lf = open(testlistfile,'r')
lines = lf.readlines()
filelist =[file.rstrip('\n')for file in lines]
lf.close()
clables =  open(txt_folder,'w')
lables = {
        'background':0,
        'aeroplane':1,
        'biccle':2,
        'bird':3,
        'boat':4,
        'bottle':5,
        'bus':6,
        'car':7,
        'cat':8,
        'chair':9,
        'cow':10,
        'diningtable':11,
        'dog':12,
        'horse':13,
        'motorbike':14,
        'person':15,
        'pottedplant':16,
        'sheep':17,
        'sofa':18,
        'train':19,
        'tvmonitor':20,
        }
def exist_a_class(xml_path):
    in_file = open(xml_path,'r')
    tree=ET.parse(in_file)
    root = tree.getroot()
    existance = False
    content =  root.find('filename').text.rstrip('.jpg')
    gt = [0]*21
    for obj in root.iter('name'):
        if obj.text in lables:
            gt[lables[obj.text]] = 1
    content = content+':'+str(gt)+'\n'
    clables.write(content)
    existance = True
    in_file.close()
    return existance

print('Start Counting {}!'.format(classes[0]))
print('Read xml files in xml_folder: {}'.format(xml_folder))
count = 0
for xml_id in filelist:
    if exist_a_class(xml_folder+xml_id+'.xml'):
        count += 1

    if ((count)%5000)==0:
        print('Already loaded {} xml files!'.format(count))
print('Finally loaded {} xml files'.format(count))
print('There exists {} {}s !!'.format(count,classes[0]))

