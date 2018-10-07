import os
import tensorflow.contrib.slim as slim
import logging
import tensorflow as tf
import utils
import ImageDatasetGeneration
from Data import Data
from tensorflow.python.ops import control_flow_ops
from datetime import datetime

logger = logging.getLogger('Online Hanzi recognizer')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

LOG_DIR = './log'

tf.app.flags.DEFINE_integer('evaluation_step_frequency', 100,
                            "Evaluates every 'evaluation_step_frequency' step")  # 30
tf.app.flags.DEFINE_string('mode', 'training', 'Running mode: {"training", "test"}')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')  # 20
tf.app.flags.DEFINE_integer('saving_step_frequency', 600, "Save the network every 'saving_step_frequency' steps")
tf.app.flags.DEFINE_integer('epoch', 15, 'Number of epoches')
tf.app.flags.DEFINE_integer('max_steps', 100000, 'the max number of steps for training')  # 300
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')

tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', False, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
FLAGS = tf.app.flags.FLAGS


def build_graph(top_k, images, labels=None):
    with tf.variable_scope("placeholder"):
        keep_nodes_probabilities = tf.placeholder(dtype=tf.float32, shape=[], name='keep_nodes_probabilities')
        is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')

    with tf.variable_scope("convolutional_layer"):
        # stride = 1
        conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
        max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')

        conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
        max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')

        conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
        max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')

        conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
        conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
        max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')

    flatten = slim.flatten(max_pool_4)

    with tf.variable_scope("fc_layer"):
        fc1 = slim.fully_connected(slim.dropout(flatten, keep_nodes_probabilities), 1024, activation_fn=tf.nn.relu, scope='fc1')
        logits = slim.fully_connected(slim.dropout(fc1, keep_nodes_probabilities), Data.CHARSET_SIZE, activation_fn=None, scope='fc2')

    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    with tf.variable_scope("accuracy"):
        math_ops = tf.cast(tf.argmax(logits, 1), tf.int32)
        tensor_flag = tf.equal(math_ops, labels)
        accuracy = tf.reduce_mean(tf.cast(tensor_flag, tf.float32))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)

    step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate =2e-4, global_step=step, decay_rate=0.97, decay_steps=2000, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # the step will be incremented after the call to optimizer.minize()
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=step)
    probabilities = tf.nn.softmax(logits)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    #accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_nodes_probabilities': keep_nodes_probabilities,
            'step': step,
            "optimizer": optimizer,
            'loss': loss,
            'accuracy': accuracy,
            'predicted_val_top_k': predicted_val_top_k,
            'predicted_index_top_k': predicted_index_top_k,
            'merged_summary_op': merged_summary_op,
            'is_training': is_training,
            'train_op': train_op
            }


def training():
    training_data = Data(data_dir=Data.DATA_TRAINING)
    test_data = Data(data_dir=Data.DATA_TEST)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True,
                                          log_device_placement=True)) as sess:

        training_init_op = training_data.get_batch(batch_size=FLAGS.batch_size, aug=True)
        next_training_sample = training_data.get_next_element()
        graph = build_graph(top_k=1, images=next_training_sample[0], labels=next_training_sample[1])
        sess.run(tf.global_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
        saver = tf.train.Saver()
        train_writer = tf.summary.FileWriter(LOG_DIR + '/training', sess.graph)
        test_writer = tf.summary.FileWriter(LOG_DIR + '/test')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint %s" % ckpt)
                start_step += int(ckpt.split('-')[-1])

        logger.info('Start training')
        logger.info("Training data size: %d" % training_data.size)
        logger.info("Test data size: %d" % test_data.size)
        print("Getting training data...")

        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(training_init_op)

        def train():
            start_time = datetime.now()
            print("Getting training data took %s " % utils.r(start_time))
            feed_dict = {graph['keep_nodes_probabilities']: 0.8, graph['is_training']: True}
            _, loss, train_summary, step = sess.run(
                [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['step']],
                feed_dict=feed_dict)
            train_writer.add_summary(train_summary, step)
            logger.info("Step #%s took %s. Loss: %d \n" % (step, utils.r(start_time), loss))
            return step

        start_time = datetime.now()
        eval_frequency = FLAGS.evaluation_step_frequency
        while not coordinator.should_stop():
            step = train()
            if step > FLAGS.max_steps:
                break

            if (step % eval_frequency == 0) and (step >= eval_frequency):
                feed_dict = {graph['keep_nodes_probabilities']: 1.0, graph['is_training']: False}
                accuracy_test, test_summary = sess.run([graph['accuracy'], graph['merged_summary_op']],
                                                       feed_dict=feed_dict)
                test_writer.add_summary(test_summary, step)
                logger.info('---------- Step #%d   Test accuracy: %.2f ' % (int(step), accuracy_test))
            if step % FLAGS.saving_step_frequency == 1:
                logger.info('Saving checkpoint of step %s' % step)
                saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'online_hanzi_recog'),
                           global_step=graph['step'])

        logger.info('Training Completed in  %s ' % utils.r(start_time))
        coordinator.request_stop()
        train_writer.close()
        test_writer.close()
        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'online_hanzi_recog'), global_step=graph['step'])
        coordinator.join(threads)
        sess.close()


def recognize_image(image):
    recognizer = get_predictor()
    return recognizer(image)


def get_predictor():
    def recognizer(input_image):
        sess = tf.Session()
        data = Data(image_file_name=input_image)
        training_init_op = data.get_batch(batch_size=1, aug=True)
        training_sample = data.get_next_element()
        graph = build_graph(top_k=3, images=training_sample[0], labels=training_sample[1])
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        sess.run(training_init_op)
        predicted_probabilities, predicted_indexes = sess.run(
            [graph['predicted_val_top_k'], graph['predicted_index_top_k']],
            feed_dict={graph['keep_nodes_probabilities']: 1.0, graph['is_training']: False})

        cols, rows = predicted_indexes.shape
        list_length = rows if (rows < 6) else 6
        predicted_indexes2 = predicted_indexes[0, :list_length]
        label_char_dico = Data.get_label_char_dico('../' + ImageDatasetGeneration.CHAR_LABEL_DICO_FILE_NAME)
        predicted_chars = [label_char_dico.get(index) for index in predicted_indexes2]
        predicted_probabilities2 = ["%.1f" % (proba * 100) for proba in predicted_probabilities[0, :list_length]]
        return predicted_chars, predicted_indexes2, predicted_probabilities2

    return recognizer


def main(_):
    print("Mode: %s " % FLAGS.mode)
    if FLAGS.mode == 'recognize_image':
        image_path = Data.DATA_ROOT_DIR + '/test/00000/34385.png'
        probability, label, character = recognize_image(image_path)
        print('Label: %s  Probability: %f  Character: %s' % (label, probability * 100, character))

    elif FLAGS.mode == "training":
        training()


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')

    tf.app.run()
else:  # webServer mode
    tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoint/', 'the checkpoint dir')
