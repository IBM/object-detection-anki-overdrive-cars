# from https://github.com/tensorflow/models/tree/master/research/object_detection/dataset_tools
# and https://gist.github.com/saghiralfasly/ee642af0616461145a9a82d7317fb1d6
 
import tensorflow as tf
from object_detection.utils import dataset_util
import os
import io
import hashlib
import xml.etree.ElementTree as ET
import random
from PIL import Image

def create_example(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        image_name = root.find('filename').text
        file_name = image_name.encode('utf8')
        size=root.find('size')
        width = int(size[0].text)
        height = int(size[1].text)
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        for member in root.findall('object'):
           classes_text.append(member[0].text)

           def class_text_to_int(row_label):
              if row_label == 'car-red':
                 return 1
              if row_label == 'car-blue':
                 return 2
              if row_label == 'phone':
                 return 3

           classes.append(class_text_to_int(member[0].text))

           xmin.append(float(member[4][0].text) / width)
           ymin.append(float(member[4][1].text) / height)
           xmax.append(float(member[4][2].text) / width)
           ymax.append(float(member[4][3].text) / height)
           difficult_obj.append(0)
           truncated.append(0)
           poses.append('Unspecified'.encode('utf8'))

        full_path = os.path.join('./data/images', '{}'.format(image_name))
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        if image.format != 'JPEG':
           raise ValueError('Image format not JPEG')
        key = hashlib.sha256(encoded_jpg).hexdigest()
		
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(file_name),
            'image/source_id': dataset_util.bytes_feature(file_name),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))	
        return example	
		
def main(_):
    writer_train = tf.python_io.TFRecordWriter('./data/train.record')     
    writer_test = tf.python_io.TFRecordWriter('./data/test.record')
    filename_list=tf.train.match_filenames_once("./data/annotations/*.xml")
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess=tf.Session()
    sess.run(init)
    list=sess.run(filename_list)
    random.shuffle(list)  
    i=1 
    tst=0
    trn=0 
    for xml_file in list:
      example = create_example(xml_file)
      if (i%5)==0: 
         writer_test.write(example.SerializeToString())
         tst=tst+1
      else:        
         writer_train.write(example.SerializeToString())
         trn=trn+1
      i=i+1
      print(xml_file)
    writer_test.close()
    writer_train.close()
    print('Successfully converted dataset to TFRecord.')
    print('training dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)	
	
if __name__ == '__main__':
    tf.app.run()