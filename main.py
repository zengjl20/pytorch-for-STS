#! /usr/bin/env python
import os
import argparse
import datetime
import torch
import torchtext.legacy.data as data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import pdb


parser = argparse.ArgumentParser(description='STS_model')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=128, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=1000, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1500, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# cnn model
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# lstm model
parser.add_argument('-hid-dim', type=int, default=256, help='number of hidden dimension of lstm[default: 128]')
parser.add_argument('-num-layers', type=int, default=1, help='number of layers of lstm')
parser.add_argument('-num-dir', type=int, default=2, help='number of directions of lstm')
parser.add_argument('-use-att', type=bool, default=False, help='whether to use self attention')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-use-w2v', type=bool, default=True, help='whether to use pretrained glove')
parser.add_argument('-use-cnn', type=bool, default=True, help='if True then use cnn, otherwise use lstm')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()


# load dataset
def mr(text_field, label_field, **kargs):
    train_data = mydatasets.MR.splits(text_field, label_field)
    dev_data = mydatasets.MR.splits(text_field, label_field, flag='dev')
    test_data = mydatasets.MR.splits(text_field, label_field, flag='test')
    text_field.build_vocab(train_data, dev_data, test_data, vectors='glove.6B.100d')
    #label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter, test_iter = data.Iterator.splits(
                                (train_data, dev_data, test_data), 
                                batch_sizes=(args.batch_size, len(dev_data), len(test_data)),
                                **kargs)
    return train_iter, dev_iter, test_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
train_iter, dev_iter, test_iter = mr(text_field, label_field, device=-1, repeat=False)


# update args and print
args.embed_num = len(text_field.vocab)
#args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
if args.use_cnn:
    sts_model = model.CNN_Text(args)
else:
    sts_model = model.LSTM_Text(args)
if args.use_w2v:
    sts_model.embed.weight.data.copy_(text_field.vocab.vectors)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    sts_model.load_state_dict(torch.load(args.snapshot))
if args.cuda:
    torch.cuda.set_device(args.device)
    sts_model = sts_model.cuda()        

# train or test
if args.predict is not None:
    predict = train.predict(args.predict, sts_model, text_field, args.cuda)
    print('\n[Text]  {}\nSimilarity: {}\n'.format(args.predict, predict))
elif args.test:
    try:
        train.eval(test_iter, sts_model, args) 
    except Exception as e:
        print("\nSorry. The snapshot doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, sts_model, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

