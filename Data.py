import random
import os
import tensorflow as tf
import utils
import codecs
import numpy
from datetime import datetime


class Data:

    # DATA_ROOT_DIR = '/DATA/CASIA/onlineHanziRecognizer'
    DATA_ROOT_DIR = '/TEMP_DATA_SET'
    DATA_TRAINING = DATA_ROOT_DIR + '/training'
    DATA_TEST = DATA_ROOT_DIR + '/test'
    CHARSET_SIZE = 3755
    IMAGE_SIZE = 64

    def __init__(self, data_dir= None, image_file_name = None, random_flip_up_down=False, random_brightness=False, random_contrast=True):

        self.random_flip_up_down = random_flip_up_down
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast

        if (image_file_name):# add comment
            self.image_file_paths = [image_file_name]
            self.labels = numpy.array([0])
            return

        truncate_path = data_dir + os.sep + ('%05d' % Data.CHARSET_SIZE)  # display number with 5 leading 0
        self.image_file_paths = []
        for root, sub_folder, image_file_names in os.walk(data_dir):
            if root < truncate_path:
                self.image_file_paths += [os.path.join(root, image_file_name) for image_file_name in image_file_names]
        random.shuffle(self.image_file_paths)
        # the labels are the name of directories converted to int: {'00000', '00001', '00002', ...}
        self.labels = []
        for image_file_path in self.image_file_paths:
            # images_dir_name example : '00000', '00001', '00002'
            images_dir_name = image_file_path[len(data_dir) + 1:].split(os.sep)[0]
            img_dir_name = int(images_dir_name)
            self.labels.append(img_dir_name)
        print("self.labels size: %d" % len(self.labels))


    @staticmethod
    def augmentation(self, images):

        if self.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if self.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        elif self.random_contrast:
            images = tf.image.random_contrast(images, 0.9, 1.1)  #  0.8 1.2
        return images

    @staticmethod
    def get_label_char_dico(file):

        path = os.getcwd() + os.sep + file
        char_label_dictionary = Data.load_char_label_dico(path)
        label_char_dico = Data.get_label_char_map(char_label_dictionary)
        return label_char_dico

    def get_label_char_map(character_label_dico):
        inverted_map = {v: k for k, v in character_label_dico.items()}
        return inverted_map

    @property
    def size(self):
        return len(self.labels)

    def get_batch(self, batch_size, aug=False):

        def _parse_function(filename):
            # convert to grey
            image = tf.read_file(filename)
            image_grey = tf.image.convert_image_dtype(tf.image.decode_png(image, channels=1), tf.float32)
            if aug:
                image_grey = self.augmentation(self, image_grey)

            # standardize the image size .
            standard_size = tf.constant([Data.IMAGE_SIZE, Data.IMAGE_SIZE], dtype=tf.int32)
            images = tf.image.resize_images(image_grey, standard_size)
            return images

        image_file_path_dataset = tf.data.Dataset.from_tensor_slices(self.image_file_paths)
        label_dataset = tf.data.Dataset.from_tensor_slices(self.labels)
        image_file_path_dataset = image_file_path_dataset.map(_parse_function)

        # zip the x and y training data together and shuffle, batch etc.
        dataset = tf.data.Dataset.zip((image_file_path_dataset, label_dataset)).shuffle(500).repeat().batch(batch_size)

        self.iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        training_init_op = self.iterator.make_initializer(dataset)
        return training_init_op

    def get_next_element(self):
        next_element = self.iterator.get_next()
        return next_element

    def load_char_label_dico(filePath):

        print("Loading CharLabelMap ... ")
        start_time = datetime.now()
        charLabelMap = {}
        with codecs.open(filePath, 'r', 'gb2312') as f:
            for line in f:
                lineWithoutCR = line.split("\n")[0]
                splitted = lineWithoutCR.split(" ")
                char = splitted[0]
                label = int(splitted[1])
                charLabelMap[char] = label
        print("Execution time: %s s." % utils.r(start_time))
        return charLabelMap