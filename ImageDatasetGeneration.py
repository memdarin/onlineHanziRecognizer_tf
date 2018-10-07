import os
import numpy as np
import cv2
import argparse
import random
import sys
import pickle
import struct
import codecs
import time;
import utils;
import Data

CHAR_LABEL_DICO_FILE_NAME = 'charLabelDicoFile.txt'
GNT_TRAINING_PATH = 'C:\DATA\PROJECTS\CASIA\OFFLINE\HWDB1.1trn_gnt'
GNT_TEST_PATH = 'C:\DATA\PROJECTS\CASIA\OFFLINE\HWDB1.1tst_gnt'
OUTPUT_DIR = 'C:\TEMP_GENERATED_DATASET'


# returns a map . key = chinese character, value = index
#  {'一' :0, '丁' : 1, '七' : 2, ...}
def build_char_index_dictionary():

    print("Building dictionary... ")
    start_time = time.time()
    dico_file = codecs.open(CHAR_LABEL_DICO_FILE_NAME, 'w', 'gb2312')
    character_set = set()
    for file_name in os.listdir(GNT_TRAINING_PATH):
        file_path = os.path.join(GNT_TRAINING_PATH, file_name)
        f = open(file_path, 'r')
        for _, tag_code in extract_image_and_tag_from_gnt_file(f):
            uni = struct.pack('>H', tag_code).decode('gb2312')
            character_set.add(uni)

    characters = list(character_set)
    character_index_dico = dict(zip(sorted(characters), range(len(characters))))
    for character, index in character_index_dico.items():
        dico_file.write(character + " " + str(index) + "\n")
    dico_file.close()
    print("Total %s characters. Execution time: %d s." % (str(len(character_index_dico)), time.time() - start_time))
    return character_index_dico

#  Extracts all the character images contained in one gnt file and put each extracted
# image into its corresponding directory.
def convert_gnt_to_png(gnt_dir, png_dir, char_label_dico):

    start_time = time.time()
    i = 0
    for file_name in os.listdir(gnt_dir):
        file_path = os.path.join(gnt_dir, file_name)
        gnt_file = open(file_path, "r")
        for image, tag_code in extract_image_and_tag_from_gnt_file(gnt_file):
            i += 1
            tag_code_uni = struct.pack('>H', tag_code).decode('gb2312') # chinese character
            character_dir = png_dir + "/" + '%0.5d' % char_label_dico[tag_code_uni]
            # character_dir examples : '00000', '00001', '00002'...
            # character_dir is a dir that contains all the 240 images of a given character
            os.makedirs(character_dir, exist_ok=True)
            image_name = str(i) + ".png"
            cv2.imwrite(character_dir + '/' + image_name, image)
        gnt_file.close()
    print("Execution time: %d s." %  time.time() - start_time)
    return i

def extract_image_and_tag_from_gnt_file(file):

    while True:
        header = np.fromfile(file, dtype="uint8", count=10)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tag_code = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        image = np.fromfile(file, dtype='uint8', count=width * height).reshape((height, width))
        yield image, tag_code

def main():

    training_out_path = os.path.join(OUTPUT_DIR, "training")
    test_out_path = os.path.join(OUTPUT_DIR, "test")

    #char_dictionary = build_char_index_dictionary()
    char_label_dictionary = Data.load_char_label_map(CHAR_LABEL_DICO_FILE_NAME)

    print("Extracting training images.. ")
    training_images = convert_gnt_to_png(GNT_TRAINING_PATH, training_out_path, char_label_dictionary)
    print("Total " + str(training_images) + " images in training set.")

    print("Extracting test images.. ")
    test_images = convert_gnt_to_png(GNT_TEST_PATH, test_out_path, char_label_dictionary)
    print("Total " + str(test_images) + " images in test set.")

if __name__ == '__main__':
    main()
