import tensorflow as tf
import sys, os
from model import Model
from dataset import Dataset
from network import *
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tempfile import TemporaryFile

import struct # for lazy load

def main(path):
    # Load dataset
    dataset = Dataset(path)

    # Learning params
    learning_rate = 0.002
    batch_size = 48
    training_iters = 2000
    display_step = 10
    test_step = 200
    save_step = 250


    # Network params
    n_classes = 24
    keep_rate = 0.5 # for dropout

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    # Model
    pred = Model.alexnet(x, keep_var) # definition of the network architecture

    # Loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # Evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Init
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    logFileTrain = open('lr'+str(learning_rate)+'_train_f.log', 'w')
    logFileTest = open('lr'+str(learning_rate)+'test_f.log', 'w')


    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        # Load pretrained model
        load_with_skip('pretrained_alexnet.npy', sess, ['fc7', 'fc8'])

        print('Start training.')
        step = 1
        while step < training_iters:
            batch_xs, batch_ys = dataset.next_batch(batch_size, 'train')

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})

            # Display testing status
            if step%test_step == 0:
                test_acc = 0.
                test_count = 0
                for _ in range(int(dataset.test_size/batch_size)+1): # test accuracy by group of batch_size images
                    batch_tx, batch_ty = dataset.next_batch(batch_size, 'test')
                    # print(batch_tx.shape)
                    acc = sess.run(accuracy, feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                    test_acc += acc
                    test_count += 1
                test_acc /= test_count
                print("{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc))
                print("Iter {}: Acc {:.4f}".format(step, test_acc), file=logFileTest)
                logFileTest.flush()

            # Display training status
            if step%display_step == 0:
                acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.}) # Training-accuracy
                batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.}) # Training-loss
                print("{} Iter {}: Training Loss = {:.4f}, Accuracy = {:.4f}".format(datetime.now(), step, batch_loss, acc))
                print("Iter {}: Loss {:.4f}".format(step, batch_loss), file=logFileTrain)
                print("Iter {}: Acc {:.4f}".format(step, acc), file=logFileTrain)
                logFileTrain.flush()

            # Display saving
            if step%save_step == 0:
                # print("Saving checkpoint")
                checkpoint_name = os.path.join('checkpoints/', 'f_lr0.002_model7+8_iter_'+str(step)+'.ckpt')
                save_path = saver.save(sess, checkpoint_name)
            step += 1

if __name__ == '__main__':
    main(sys.argv[1])













