import tensorflow as tf
from deployment import model_deploy
from datasets import imagenet
from model import load_batch
# from model import resnet101_2 as resnet101
from  model import resnet101
import utils, numpy as np
import tensorflow.contrib.slim as slim

flags = tf.app.flags
flags.DEFINE_string('data_dir', '../data/imagenet10k-hhd',
                    'Directory with data.')
flags.DEFINE_integer('batch_size', 32,
                     'Batch size.')
flags.DEFINE_bool('dbg', True,
                  'dbg')
flags.DEFINE_string('log_dir',
                    '../output/imagenet101',
                    'Directory with the log data.')
flags.DEFINE_integer('num_clones', 2,
                     'num_clones')
flags.DEFINE_string('checkpoint_path', '../models/resnet_v2_101.ckpt',
                    'checkpoint path')
flags.DEFINE_integer('nclasses', 1000,  # todo
                     'nclasses')
flags.DEFINE_integer('nsteps', 10000,
                     'nsteps')

beta = 0.1
gamma = 0.1
FLAGS = flags.FLAGS
if FLAGS.dbg:
    FLAGS.log_dir = utils.randomword(10)


def clone_fn(batch_queue):
    images, labels = batch_queue.dequeue()
    tf.summary.image('inputs', images)
    predictions, end_points = resnet101(images, classes=FLAGS.nclasses)

    one_hot_labels = slim.one_hot_encoding(
        labels,
        FLAGS.nclasses)
    tf.logging.info('>> dataset has class:{}'.format(FLAGS.nclasses))

    loss_100 = tf.losses.softmax_cross_entropy(
        logits=predictions,
        onehot_labels=one_hot_labels)

    # tf.losses.add_loss(loss_100)

    loss_reg = tf.reduce_sum(tf.losses.get_regularization_losses())

    tf.logging.info('loss are {}'.format(tf.losses.get_losses()))

    total_loss = tf.losses.get_total_loss()
    # todo many be add summaries afterwards
    slim.summary.scalar('loss/total', total_loss)
    slim.summary.scalar('loss/loss100', loss_100)
    slim.summary.scalar('loss/loss_reg', loss_reg)

    return end_points


def get_init_fn():
    slim.assign_from_checkpoint_fn(
        FLAGS.checkpoint_path,
        slim.get_variables_to_restore(
            exclude=[".*logits.*", ".*Ftrl.*", '.*Momentum.*', '.*fully_connected.*', '.*global_step.*']),
        ignore_missing_vars=False
        )


def train_step_fn(session, *args, **kwargs):
    from tensorflow.contrib.slim.python.slim.learning import train_step

    total_loss, should_stop = train_step(session, *args, **kwargs)

    # if train_step_fn.step % 20 == 0:
    #     acc_ = session.run(train_step_fn.acc)
    #     # acc_vall_ = []
    #     # for ind in range(10000 // FLAGS.batch_size + 1):
    #     #     acc_vall_.append(session.run(train_step_fn.acc_val))
    #
    #     print('>> Step %s - Loss: %.2f  acc: %.2f%%' % (
    #         str(train_step_fn.step).rjust(6, '0'), total_loss,
    #         # np.mean(acc_vall_) * 100,
    #         acc_ * 100))
    #
    # train_step_fn.step += 1
    return [total_loss, should_stop]


# train_step_fn.step = 0
# train_step_fn.acc_val = acc_val
# train_step_fn.acc = acc


def main(args):
    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.get_default_graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=False,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0,
        )
        with tf.device(deploy_config.variables_device()):
            global_step = slim.get_or_create_global_step()

        dataset = imagenet.get_split('train', FLAGS.data_dir)

        with tf.device(deploy_config.inputs_device()):
            # load batch of dataset
            batch_queue = load_batch(
                dataset,
                FLAGS.batch_size,
                height=224,
                width=224,
                is_training=True)

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        end_points = clones[0].outputs
        # todo can add summarys here

        # todo moving avarage  for each noisy loss

        with tf.device(deploy_config.optimizer_device()):
            global_step = slim.get_or_create_global_step()
            learning_rate = tf.train.exponential_decay(
                1e-1, global_step,
                1000, 0.5, staircase=True)
            slim.summary.scalar('lr', learning_rate)

            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

        total_loss, clone_gradients = model_deploy.optimize_clones(
            clones, optimizer,
            # var_list=slim.get_trainable_variables('resnet_v2_101/block4/.*') + slim.get_trainable_variables('.*logits.*'),
        )

        grad_updates = optimizer.apply_gradients(clone_gradients, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        _sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        _sess_config.gpu_options.allow_growth = True
        slim.learning.train(
            train_tensor,
            session_config=_sess_config,
            logdir=FLAGS.log_dir,
            is_chief=1,
            init_fn=get_init_fn(),
            train_step_fn=train_step_fn,
            summary_op=summary_op,
            number_of_steps=FLAGS.nsteps,
        )


if __name__ == '__main__':
    utils.rm(FLAGS.log_dir)
    utils.init_dev(utils.get_dev(n=FLAGS.num_clones))
    tf.app.run()
