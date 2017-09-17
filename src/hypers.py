from easydict import EasyDict as edict
from utils import root_path
import utils

cifar100 = edict(data_dir='../data/cifar100',
                 batch_size=32,
                 log_dir='../output/multiloss-dry12',
                 checkpoint_path='../models/resnet50/resnet_v2_50.ckpt',
                 multiloss=False,
                 )

cifar100_eval = edict(data_dir='../data/cifar100',
                      batch_size=32,
                      log_dir='../output/multiloss-dry12',
                      checkpoint_path=cifar100.log_dir,
                      multiloss=False,
                      )

cifar10 = edict(data_dir=root_path + '/data/cifar10',
                batch_size=128,
                log_dir=root_path + '/output/cifar10',
                checkpoint_path='../models/resnet101/resnet_v2_101.ckpt',
                )

cifar10_eval = edict(
    data_dir=root_path + '/data/cifar10',
    batch_size=128,
    log_dir=root_path + '/output/cifar10_eval',
    checkpoint_dir=cifar10.log_dir,
    eval_interval_secs=60,
    num_evals=10000 // 128 + 1,
)

imagenet = edict()
imagenet.data_dir = utils.root_path + '/data/imagenet10k'
imagenet.batch_size = 32
imagenet.num_clones = 4
imagenet.checkpoint_path = utils.root_path + '/models/resnet101/resnet_v2_101.ckpt'

imagenet.nclasses = 10
imagenet.nsteps = 10000

imagenet.log_dir = '../output/' + utils.randomword(10),

imagenet.beta = 0.1
imagenet.gamma = 0.1
