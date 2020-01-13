import argparse

import tensorflow as tf

from auto_encoder import Model

tf.set_random_seed(19)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', type=str, default='train')
parser.add_argument('--dataset_name', dest='dataset_name', type=str)
args = parser.parse_args()


def main(_):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    usable_gpu = "0"
    tfconfig.gpu_options.visible_device_list = usable_gpu
    with tf.Session(config=tfconfig) as sess:
        model = Model(sess, args)
        model.train() if args.phase == 'train' else model.test()


if __name__ == '__main__':
    tf.app.run()