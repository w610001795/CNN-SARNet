import tensorflow as tf
import random
import logging
import numpy as np
from PIL import Image
#from keras.preprocessing.image import ImageDataGenerator #new library
from train import train
from validation import validation
from inference import inference
import os
from log import logger

#build the log file
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

#Flags of the main
tf.app.flags.DEFINE_boolean('flip_up_down', True, 'Whether to flip up down')
tf.app.flags.DEFINE_boolean('flip_left_right', True, 'Whether to flip right left, add flip up down')
#tf.app.flags.DEFINE_boolean('transpose_image', True, 'Whether to transpose, add flip right left and flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, 'whether to adjust brightness')
tf.app.flags.DEFINE_boolean('random_contrast', True, 'whether to random constrast')

tf.app.flags.DEFINE_integer('charset_size', 2, 'Choose your classifical size to conduct our experiment.')
tf.app.flags.DEFINE_integer('image_size', 128, 'Provide the same value in training.')
tf.app.flags.DEFINE_integer('max_steps', 120000, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 50, 'the step num to eval')
tf.app.flags.DEFINE_integer('save_steps', 2000, 'the steps to save')

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './train/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './test/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', '/./log/', 'the logging dir')
tf.app.flags.DEFINE_string('image_path', '../data/xx.png', 'the inference dataset dir')

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('batch_size', 256, 'Validation batch size')
tf.app.flags.DEFINE_boolean('dropout', 0.5, 'THe value of your dropout in training')
tf.app.flags.DEFINE_string('mode', 'train', 'Running mode - one of "train", "validation", "inference" ')
FLAGS = tf.app.flags.FLAGS

def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":

        train()

    elif FLAGS.mode == 'validation':

        dct = validation()
        result_file = 'result.dict'
        logger.info('Writing result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)
        logger.info('Writing log file ends')

    elif FLAGS.mode == 'inference':

        final_predict_val, final_predict_index = inference(FLAGS.image_path)
	    parent_path1 = os.path.dirname(FLAGS.image_path)
	    file_name = os.path.split(FLAGS.image_path)[-1]
	    class_name = os.path.split(parent_path1)[-1]
        logger.info('The result info label {0} ( {1} ) \npredict index of top k: {2} \npredict val of top k: {3}'.format(class_name, file_name, final_predict_index,
                                                                                         final_predict_val))

if __name__ == "__main__":
    tf.app.run()