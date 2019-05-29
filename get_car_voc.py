# -*- coding: utf-8 -*-
# @Function:There are 20 classes in VOC data set. If you need to extract specific classes, you can use this program to extract them.
 
import os
import shutil
ann_filepath='/datasets/VOCdevkit/VOC2012/Annotations/' #Note to change to your own address
img_filepath='/datasets/VOCdevkit/VOC2012/JPEGImages/' #Note to change to your own address
img_savepath='/datasets/VOCdevkit/VOC2012/JPEGImages_ssd/' #Note to change to your own address
ann_savepath='/datasets/VOCdevkit/VOC2012/Annotations_ssd/' #Note to change to your own address
if not os.path.exists(img_savepath):
    os.mkdir(img_savepath)
 
if not os.path.exists(ann_savepath):
    os.mkdir(ann_savepath)
names = locals()
classes = ['aeroplane','bicycle','bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow','diningtable',
           'dog', 'horse', 'motorbike', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor', 'person']
 
 
for file in os.listdir(ann_filepath):
    print(file)
    fp = open(ann_filepath + '//' + file)
    ann_savefile=ann_savepath+file
    fp_w = open(ann_savefile, 'w')
    lines = fp.readlines()
 
    ind_start = []
    ind_end = []
    lines_id_start = lines[:]
    lines_id_end = lines[:]
 
    classes1 = '\t\t<name>car</name>\n' #Write a few classes to write a few classes, I only need person
    #classes2 = '\t\t<name>motorbike</name>\n'
    #classes3 = '\t\t<name>bus</name>\n'
    #classes4 = '\t\t<name>car</name>\n'
    #classes5 = '\t\t<name>bicycle</name>\n'
 
 
         #Found the object block in xml and record it
    while "\t<object>\n" in lines_id_start:
        a = lines_id_start.index("\t<object>\n")
        ind_start.append(a)
        lines_id_start[a] = "delete"
 
 
    while "\t</object>\n" in lines_id_end:
        b = lines_id_end.index("\t</object>\n")
        ind_end.append(b)
        lines_id_end[b] = "delete"
 
         #names stores all object blocks
    i = 0
    for k in range(0, len(ind_start)):
        names['block%d' % k] = []
        for j in range(0, len(classes)):
            if classes[j] in lines[ind_start[i] + 1]:
                a = ind_start[i]
                for o in range(ind_end[i] - ind_start[i] + 1):
                    names['block%d' % k].append(lines[a + o])
                break
        i += 1
        print(names['block%d' % k])
 
 
         #xml 
    string_start = lines[0:ind_start[0]]
         #xml 
    string_end = [lines[len(lines) - 1]]
 
 
         #Search in the given class, if it exists, write the object block information
    a = 0
    for k in range(0, len(ind_start)):
        if classes1 in names['block%d' % k]:
            a += 1
            string_start += names['block%d' % k]
        
    string_start += string_end
    for c in range(0, len(string_start)):
        fp_w.write(string_start[c])
    fp_w.close()
         #If there is no module we are looking for, delete this xml, if you copy the picture
    if a == 0:
        os.remove(ann_savepath+file)
    else:
        name_img = img_filepath + os.path.splitext(file)[0] + ".jpg"
        shutil.copy(name_img, img_savepath)
    fp.close()