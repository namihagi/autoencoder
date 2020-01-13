import os
import shutil
import json
import time

import numpy as np
import tensorflow as tf
from glob import glob
from PIL import Image

from tensorflow.contrib import slim

BATCH_NORM_MOMENTUM = 0.9


def add_noise(images):
    return images


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.test_dir = './test'

        # train loop setting
        if args.phase == 'train':
            self.is_training = True
        else:
            self.is_training = False
        self.epoch = 300
        self.hidden_dim = 64
        self.lr = 0.001

        # input setting
        self.dataset_name = args.dataset_name
        self.dataset_path = os.path.join('./datasets', self.dataset_name)
        self.checkpoint_dir = os.path.join('./checkpoint', self.dataset_name)
        self.batch_size = 8
        self.image_size = 1024
        self.image_channels = 3

        self._build_model()

        # saver
        self.saver = tf.train.Saver()
        self.log_dir = './log/{}'.format(self.dataset_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def _build_model(self):
        # define input placeholder
        self.input_images = tf.placeholder(tf.float32,
                                           [self.batch_size, self.image_size, self.image_size, self.image_channels],
                                           name='input_images')
        self.real_images = tf.placeholder(tf.float32,
                                          [self.batch_size, self.image_size, self.image_size, self.image_channels],
                                          name='real_images')

        with tf.variable_scope('encoder'):
            params = {
                'padding': 'SAME',
                'activation_fn': lambda x: tf.nn.relu(x),
                'normalizer_fn': self.batch_norm, 'data_format': 'NHWC'
            }
            with slim.arg_scope([slim.conv2d], **params):
                with slim.arg_scope([slim.max_pool2d], stride=2, padding='SAME', data_format='NHWC'):
                    x = slim.conv2d(self.input_images, self.hidden_dim, (3, 3), stride=2, scope='conv1')
                    # shape is (image_size/2, image_size/2)
                    x = slim.conv2d(x, self.hidden_dim * 2, (3, 3), stride=2, scope='conv2')
                    # shape is (image_size/4, image_size/4)
                    x = slim.conv2d(x, self.hidden_dim * 3, (3, 3), stride=2, scope='conv3')
                    # shape is (image_size/8, image_size/8)
                    x = slim.conv2d(x, self.hidden_dim * 4, (3, 3), stride=2, scope='conv4')
                    # shape is (image_size/16, image_size/16)

        with tf.variable_scope('decoder'):
            params = {
                'padding': 'SAME',
                'activation_fn': lambda x: tf.nn.relu(x),
                'normalizer_fn': self.batch_norm, 'data_format': 'NHWC'
            }
            with slim.arg_scope([slim.conv2d_transpose], **params):
                x = slim.conv2d_transpose(x, self.hidden_dim * 3, (3, 3), stride=2, scope='dconv1')
                # shape is (image_size/8, image_size/8)
                x = slim.conv2d_transpose(x, self.hidden_dim * 2, (3, 3), stride=2, scope='dconv2')
                # shape is (image_size/4, image_size/4)
                x = slim.conv2d_transpose(x, self.hidden_dim, (3, 3), stride=2, scope='dconv3')
                # shape is (image_size/2, image_size/2)

            self.output_images = slim.conv2d_transpose(x, self.image_channels, (3, 3), stride=2, padding='SAME',
                                                       activation_fn=lambda x: tf.nn.tanh(x), scope='dconv4')

        # loss
        self.loss = tf.reduce_mean((self.input_images - self.output_images) ** 2)
        self.loss_sum = tf.summary.scalar('loss', self.loss)

        # get variables
        self.vars = tf.trainable_variables()
        for var in self.vars:
            print(var.name)

        # optimizer
        self.opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, var_list=self.vars)

    def train(self):
        print('start train')
        # get file list
        train_images = glob(os.path.join(self.dataset_path, 'train', '*.png'))
        # input dataset
        dataset = ImagePipeline(self.is_training, train_images, self.epoch, self.batch_size, self.image_size)
        next_el = dataset.get_next_el()

        counter = 0
        start_time = time.time()
        while 1:
            try:
                image_paths, input_images = self.sess.run(next_el)
                real_images = input_images

                if np.random.randint(10) > 4:
                    input_images = add_noise(input_images)

                loss, loss_sum = self.sess.run([self.loss, self.loss_sum],
                                                feed_dict={self.input_images: input_images,
                                                           self.real_images: real_images})
                self.writer.add_summary(loss_sum, counter)

                counter += 1
                if (counter+1) % 20 == 0:
                    print('{}/{}, loss:{}'.format(counter, ((len(train_images)//self.batch_size)+1)*self.epoch, loss))
                    # self.save_sample()

            except tf.errors.OutOfRangeError as E:
                print('\nfinished saving feature_maps as npy file.')
                break
            except Exception as E:
                print(E)
                break

        # save model when finish training
        self.save(self.checkpoint_dir, counter)
        print('the total time is %4.4f' % (time.time() - start_time))

    def test(self):
        print('start test')
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # get file list
        test_images = glob(os.path.join(self.dataset_path, 'test', '*.png'))
        # input dataset
        dataset = ImagePipeline(self.is_training, test_images, self.epoch, self.batch_size, self.image_size)
        next_el = dataset.get_next_el()

        counter = 0
        start_time = time.time()
        while 1:
            try:
                image_paths, input_images = self.sess.run(next_el)

                output_images = self.sess.run(self.output_images,
                                              feed_dict={self.input_images: input_images})

                counter += 1
                print('{}/{}'.format(counter, len(test_images)//self.batch_size))

                self.save_images(image_paths, output_images)

            except tf.errors.OutOfRangeError as E:
                print('\nfinished saving feature_maps as npy file.')
                break
            except Exception as E:
                print(E)
                break

    def save_images(self, image_paths, output_images):
        output_dir = os.path.join(self.test_dir, self.dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for path, image in zip(image_paths, output_images):
            image = np.uint8((image + 1.0)/2)
            path = str(path.decode('utf-8'))
            Image.fromarray(image).save(os.path.join(output_dir, path.split('/')[-1].replace('.png', '.jpg')))

    def save(self, checkpoint_dir, counter):
        model_name = "{}.model".format(self.dataset_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=counter)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def batch_norm(self, x):
        x = tf.layers.batch_normalization(
            x, axis=3, center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
            training=self.is_training, fused=True,
            name='batch_norm'
        )
        return x


class ImagePipeline:
    def __init__(self, is_training, file_paths, epoch, batch_size=1, image_size=512):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.image_size = image_size
        self.epoch = epoch

        dataset = tf.data.Dataset.from_tensor_slices(self.file_paths)
        dataset = dataset.map(self._read_and_resize_image, num_parallel_calls=16)
        dataset = dataset.repeat(self.epoch if is_training else 1)
        dataset = dataset.shuffle(len(self.file_paths))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=self.batch_size)
        self.iter = dataset.make_one_shot_iterator()

    def get_next_el(self):
        return self.iter.get_next()

    def _read_and_resize_image(self, image_path):
        image = tf.read_file(image_path)
        image = tf.image.decode_image(image, channels=3)
        image.set_shape([None, None, 3])
        image = tf.image.resize_images(image, [self.image_size, self.image_size])
        image = tf.cast(image, tf.float32)
        image = image/127.5 - 1.0
        return image_path, image


def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    usable_gpu = "0"
    tfconfig.gpu_options.visible_device_list = usable_gpu
    with tf.Session(config=tfconfig) as sess:
        model = Model(sess)
        model.train()
        # model.test()


if __name__ == '__main__':
    tf.app.run()
