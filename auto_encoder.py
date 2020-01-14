import os
import time
from glob import glob

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from PIL import Image

from copy import deepcopy

BATCH_NORM_MOMENTUM = 0.9


def add_noise(images):
    for image in images:
        # if np.random.randint(10) > 4:
        w = np.random.randint(30, 256 - 30 - 25)
        h = np.random.randint(30, 256 - 30 - 25)
        noise = np.full((30, 30, 3), -1).astype(np.float32)
        image[h:h + 30, w:w + 30] = noise.copy()
    return images


class Model(object):
    def __init__(self, sess, args):
        self.sess = sess

        # train loop setting
        if args.phase == 'train':
            self.is_training = True
        else:
            self.is_training = False
        self.epoch = 100
        self.hidden_dim = 4
        self.lr = 0.001

        # input setting
        self.dataset_name = args.dataset_name
        self.dataset_path = os.path.join('./datasets', self.dataset_name)
        self.checkpoint_dir = os.path.join('./checkpoint', self.dataset_name)
        if args.sub_dirname is not None:
            self.checkpoint_dir += args.sub_dirname
        self.batch_size = 8
        self.image_size = 256
        self.image_channels = 3

        # output setting
        self.test_dir = os.path.join('./test', self.dataset_name)
        if args.sub_dirname is not None:
            self.test_dir += args.sub_dirname

        self._build_model()

        # saver
        self.saver = tf.train.Saver()
        self.log_dir = './log/{}'.format(self.dataset_name)
        if args.sub_dirname is not None:
            self.log_dir += args.sub_dirname
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

        # encoder
        net = tf.layers.conv2d(self.input_images, self.hidden_dim, [5, 5], strides=2, padding='SAME',
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net, self.hidden_dim * 2, [5, 5], strides=2, padding='SAME',
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net, self.hidden_dim * 3, [5, 5], strides=2, padding='SAME',
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net, self.hidden_dim * 4, [5, 5], strides=4, padding='SAME',
                               activation=tf.nn.leaky_relu)

        # decoder
        net = tf.layers.conv2d_transpose(net, self.hidden_dim * 3, [5, 5], strides=4, padding='SAME',
                                         activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d_transpose(net, self.hidden_dim * 2, [5, 5], strides=2, padding='SAME',
                                         activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d_transpose(net, self.hidden_dim, [5, 5], strides=2, padding='SAME',
                                         activation=tf.nn.leaky_relu)
        self.output_images = tf.layers.conv2d_transpose(net, 3, [5, 5], strides=2,
                                                        padding='SAME', activation=tf.nn.tanh)

        # loss
        # self.loss = tf.reduce_mean(tf.abs(self.real_images - self.output_images))
        # self.loss = tf.reduce_mean(tf.square(self.real_images - self.output_images))
        self.loss = tf.reduce_mean(tf.pow(self.real_images - self.output_images, 2))
        self.loss_sum = tf.summary.scalar('loss', self.loss)

        # get variables
        self.vars = tf.trainable_variables()
        for var in self.vars:
            print(var.name)

        # optimizer
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self):
        print('start train')
        # get file list
        train_images = glob(os.path.join(self.dataset_path, 'train', '*.png'))
        # input dataset
        with tf.device('/cpu:0'):
            # (self, is_training, file_paths, epoch, batch_size=1, image_size=512):
            dataset = ImagePipeline(self.is_training, file_paths=train_images, epoch=self.epoch,
                                    batch_size=self.batch_size, image_size=self.image_size)
            next_el = dataset.get_next_el()

        counter = 0
        start_time = time.time()
        while 1:
            try:
                image_paths, input_images = self.sess.run(next_el)
                real_images = deepcopy(input_images)
                input_images = add_noise(input_images)
                # self.save_images_real(image_paths, real_images)
                # self.save_images_input(image_paths, input_images)

                _, loss, loss_sum = self.sess.run([self.opt, self.loss, self.loss_sum],
                                                  feed_dict={self.input_images: input_images,
                                                             self.real_images: real_images})
                self.writer.add_summary(loss_sum, counter)

                counter += 1
                print('{}/{}, loss:{}'.format(counter, (len(train_images) * self.epoch) // self.batch_size, loss))
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
                # input_images = add_noise(input_images)
                self.save_images_input(image_paths, input_images)

                output_images = self.sess.run(self.output_images,
                                              feed_dict={self.input_images: input_images})
                counter += 1
                print('{}/{}'.format(counter, len(test_images) // self.batch_size))

                self.save_images(image_paths, output_images)

                for image_path, input_image, output_image in zip(image_paths, input_images, output_images):
                    print(image_path)
                    print(((input_image - output_image) ** 2).sum())

            except tf.errors.OutOfRangeError as E:
                print('\nfinished saving feature_maps as npy file.')
                break
            except Exception as E:
                print(E)
                break

        print('train image')
        # get file list
        train_images = glob(os.path.join(self.dataset_path, 'train', '*.png'))
        # input dataset
        dataset = ImagePipeline(self.is_training, train_images, self.epoch, self.batch_size, self.image_size)
        next_el = dataset.get_next_el()

        counter = 0
        while 1:
            try:
                image_paths, input_images = self.sess.run(next_el)
                # input_images = add_noise(input_images)
                self.save_images_input(image_paths, input_images)

                output_images = self.sess.run(self.output_images,
                                              feed_dict={self.input_images: input_images})
                counter += 1
                print('{}/{}'.format(counter, len(test_images) // self.batch_size))

                self.save_images(image_paths, output_images)

                if counter == 2:
                    break
                for image_path, input_image, output_image in zip(image_paths, input_images, output_images):
                    print(image_path)
                    print(((input_image - output_image) ** 2).sum())

            except tf.errors.OutOfRangeError as E:
                print('\nfinished saving feature_maps as npy file.')
                break
            except Exception as E:
                print(E)
                break

    def save_images(self, image_paths, output_images):
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        for path, image in zip(image_paths, output_images):
            image = (image + 1.0) * 127.5
            path = str(path.decode('utf-8'))
            Image.fromarray(image.astype(np.uint8)).save(
                os.path.join(self.test_dir, path.split('/')[-1].replace('.png', '.jpg')))

    def save_images_real(self, image_paths, output_images):
        real_dir = self.test_dir.replace('test', 'real')
        if not os.path.exists(real_dir):
            os.makedirs(real_dir)

        for path, image in zip(image_paths, output_images):
            image = (image + 1.0) * 127.5
            path = str(path.decode('utf-8'))
            Image.fromarray(image.astype(np.uint8)).save(
                os.path.join(real_dir, path.split('/')[-1].replace('.png', '.jpg')))

    def save_images_diff(self, image_paths, output_images):
        real_dir = self.test_dir.replace('test', 'diff')
        if not os.path.exists(real_dir):
            os.makedirs(real_dir)

        for path, image in zip(image_paths, output_images):
            image = (image + 1.0) * 127.5
            path = str(path.decode('utf-8'))
            Image.fromarray(image.astype(np.uint8)).save(
                os.path.join(real_dir, path.split('/')[-1].replace('.png', '.jpg')))

    def save_images_input(self, image_paths, output_images):
        input_dir = self.test_dir.replace('test', 'input')
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)

        for path, image in zip(image_paths, output_images):
            image = (image + 1.0) * 127.5
            path = str(path.decode('utf-8')).replace('test', 'input')
            Image.fromarray(image.astype(np.uint8)).save(
                os.path.join(input_dir, path.split('/')[-1].replace('.png', '.jpg')))

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
        image = image / 127.5 - 1.0
        return image_path, image
