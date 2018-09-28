import tensorflow as tf
from DI import DataIterator
from cnnnet import build_graph
from log import logger
import time
import os

FLAGS = tf.app.flags.FLAGS



def train():

    train_feeder = DataIterator(data_dir='./train/')
    test_feeder = DataIterator(data_dir='./test/')

    with tf.Session() as sess:

        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)

        graph = build_graph(top_k=1)

        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')

        start_step = 0

        if FLAGS.restore:

            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info('============== Start Training ==============')

        try:
            while not coord.should_stop():

                start_time = time.time()

                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: FLAGS.keep_prob}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                
                train_writer.add_summary(train_summary, step)

                end_time = time.time()

                logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))

                if step > FLAGS.max_steps:

                    break

                if step % FLAGS.eval_steps == 1:

                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])

                    feed_dict = {graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0}
                    accuracy_test, test_summary = sess.run(
                        [graph['accuracy'], graph['merged_summary_op']],
                        feed_dict=feed_dict)
                    
                    test_writer.add_summary(test_summary, step)

                    logger.info('============== Eval a batch ==============')
                    logger.info('the step {0} test accuracy: {1}'
                                .format(step, accuracy_test))
                    logger.info('============== Eval a batch ==============')

                if step % FLAGS.save_steps == 1:

                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                               global_step=graph['global_step'])
                    
        except tf.errors.OutOfRangeError:

            logger.info('============== Train Finished ==============')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
            
        finally:
            coord.request_stop()
        coord.join(threads)
