
import sys
import os
import shutil
import numpy as np
import json
import xml.etree.ElementTree as ET
import random

# your label id
START_BOUNDING_BOX_ID = 1
# PRE_DEFINE_CATEGORIES = {"Insect_infestation": 0,"Moulds": 1,"Biological_exclusion": 2,"Brown_spots": 3,"Water_Stain": 4}
# PRE_DEFINE_CATEGORIES = {"A1": 0,"A2": 1,"A3": 2,"B4": 3,"B5": 4,"C6": 5,
#                         "C7": 6,"C8": 7,"C9": 8,"C10": 9,"C11": 10,"C12": 11,
#                         "D13": 12,"D14": 13,"D15": 14,"E16": 15,
#                         "E17": 16,"E18": 17,"F19": 18,"F20": 19,"G21": 20,"G22": 21,"X": 22,"Y": 23}
PRE_DEFINE_CATEGORIES = {"stas": 0}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


# get image id
def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]

 
        filename = filename.lstrip('class')
        #filename = filename.lstrip('image_')
        return int(filename)
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.'%(filename))


def convert(xml_list, xml_dir, json_file):
    '''
    :param xml_list: the xml list which you want to transform
    :param xml_dir: your xml source dir
    :param json_file: export json dir
    :return: None
    '''
    list_fp = xml_list
    # create json templet
    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    image_id = 0
    for line in list_fp:
        line = line.strip()
        print("buddy~ Processing {}".format(line))
        # anaysis xml
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        # get image name
        if len(path) == 1:
            #filename = os.path.basename(path[0].text)
            filename = get_and_check(root, 'filename', 1).text
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        filename = line.replace('xml','jpg')
        # image_id = get_filename_as_int(filename)  # image ID
        image_id = image_id+1
        size = get_and_check(root, 'size', 1)
        # your image detail
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        # process all box or seg
        for obj in get(root, 'object'):
            # get your class_name
            category = get_and_check(obj, 'name', 1).text
            # upload your categories dict
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            annotation = dict()
            annotation['area'] = o_width*o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # segmentation
            annotation['segmentation'] = [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # write categories dict
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # export json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    root_path = os.getcwd()
    xml_train_dir = os.path.join('D:/JIM/CenterNet2-master/projects/CenterNet2/datasets/flower/xml-train/')
    xml_val_dir = os.path.join('D:/JIM/CenterNet2-master/projects/CenterNet2/datasets/flower/xml-val/')
    # xml_val_labels.remove('desktop.ini')
    xml_train_labels = os.listdir(os.path.join(xml_train_dir))
    xml_val_labels = os.listdir(os.path.join(xml_val_dir))
    # split_point = int(len(xml_train_labels)/10)
    # create coco annotations dir
    json_file_train = 'D:/JIM/CenterNet2-master/projects/CenterNet2/datasets/flower/annotations/instances_train2017.json'
    json_file_test = 'D:/JIM/CenterNet2-master/projects/CenterNet2/datasets/flower/annotations/instances_val2017.json'
    # your original image dir
    all_img_dir = 'D:/JIM/CenterNet2-master/projects/CenterNet2/datasets/flower/img/'
    # put original image to coco/train or val
    coco_img_train = 'D:/JIM/CenterNet2-master/projects/CenterNet2/datasets/flower/train2017/'
    coco_img_test = 'D:/JIM/CenterNet2-master/projects/CenterNet2/datasets/flower/val2017/'
    # split_point = split_point * 7
    # np.random.seed(100)
    # np.random.shuffle(xml_train_labels)
    #print(xml_labels)
    #xml_labels
    # train data
    xml_list = xml_train_labels[0:]
    convert(xml_list, xml_train_dir, json_file_train)
    for xml_file in xml_list:
        img_name = xml_file[:-4] + '.jpg'
        shutil.copy(os.path.join(all_img_dir, img_name),
                    os.path.join(coco_img_train, img_name))
    #validation data
    xml_list_val = xml_val_labels[0:]
    convert(xml_list_val, xml_val_dir, json_file_test)
    for xml_file in xml_list_val:
       img_name = xml_file[:-4] + '.jpg'
       shutil.copy(os.path.join(all_img_dir, img_name),
                   os.path.join(coco_img_test, img_name))

