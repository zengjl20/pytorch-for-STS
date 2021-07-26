import re
import os
import random
import tarfile
import urllib
from torchtext.legacy import data
import csv
import torch


class TarDataset(data.Dataset):
    """Defines a Dataset loaded from a downloadable tar archive.

    Attributes:
        url: URL where the tar archive can be downloaded.
        filename: Filename of the downloaded tar archive.
        dirname: Name of the top-level directory within the zip archive that
            contains the data files.
    """

    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                tfile.extractall(root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz'
    filename = 'Stsbenchmark.tar.gz'
    dirname = 'stsbenchmark'

    @staticmethod
    def sort_key(ex):
        return len(ex.text1)

    def __init__(self, text_field, label_field, path=None, examples=None, flag=None, **kwargs):
        """Create an MR dataset instance given a path and fields.

        Arguments:
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            path: Path to the data file.
            examples: The examples contain all the data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.tokenize = lambda x: clean_str(x).split()
        fields = [('text1', text_field), ('text2', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            if flag == 'dev':
                f = csv.reader(open(os.path.join(path, 'sts-dev.csv'), errors='ignore'))
            elif flag == 'test':
                f = csv.reader(open(os.path.join(path, 'sts-test.csv'), errors='ignore'))
            else:
                f = csv.reader(open(os.path.join(path, 'sts-train.csv'), errors='ignore'))
            for line in f:
                line = line[0].split('\t')
                try:
                    examples += [data.Example.fromlist([line[5], line[6], line[4]], fields)]
                except:
                    pass
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, shuffle=True, root='.', **kwargs):
        """Create dataset objects for splits of the MR dataset.

        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            shuffle: Whether to shuffle the data before split.
            root: The root directory that the dataset's zip archive will be
                expanded into; therefore the directory in whose trees
                subdirectory the data files will be stored.
            train: The filename of the train data. Default: 'train.txt'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)

        return cls(text_field, label_field, examples=examples)

if __name__=="__main__":
    def mr(text_field, label_field, **kargs):
        train_data = MR.splits(text_field, label_field)
        dev_data = MR.splits(text_field, label_field, flag='dev')
        test_data = MR.splits(text_field,label_field, flag='test')
        text_field.build_vocab(train_data, dev_data, test_data, vectors='glove.6B.100d')
        #label_field.build_vocab(train_data, dev_data)
        train_iter, dev_iter, test_iter= data.Iterator.splits(
                                    (train_data, dev_data, test_data),
                                    batch_sizes=(64, len(dev_data), len(test_data)),
                                    **kargs)
        return train_iter, dev_iter, test_iter
    print("\nLoading data...")
    text_field = data.Field(fix_length=30, lower=True)
    label_field = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
    train_iter, dev_iter, test_iter = mr(text_field, label_field, device=-1, repeat=False)
    for batch in dev_iter:
        feature = batch.text1
        logit = batch.label
        feature.t_()
        print(feature[:10])
        print('label:{}'.format(logit))
        print('label.shape:{}'.format(logit.shape))
        break
    print('len(text_vocab):{}'.format(len(text_field.vocab)))
    print('vocab:{}'.format(text_field.vocab.itos[:50]))

