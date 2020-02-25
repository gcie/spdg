import numpy
import torch
from torchvision import datasets, transforms

from .ngram import Ngram


def categorical(ngram, num_samples):
    """Sample n-gram entries with specified probabilities.

    Equivalent to `spdg.datasets.target_sequences(ngram, num_samples, ngram.n)`.

    ### Parameters:
    - ngram
    - num_samples

    ### Returns:
    - num_samples x ngram.n numpy array
    """
    return numpy.asarray([ngram.norm().sample() for _ in range(num_samples)])


def target_sequences(ngram, num_samples, sequence_length):
    """Generate sequences from given n-gram distribution.

    WARNING: Output n-gram distribution might differ (in some cases very significantly) from input!

    ### Parameters:
    - ngram (`spdg.ngram.Ngram`)
    - num_samples (integer)
    - sequence_length (integer)

    ### Returns:
    - num_samples x sequence_length numpy array
    """
    if not isinstance(ngram, Ngram):
        raise ValueError(f"ngram should inherit `spdg.ngram.Ngram`, but got {type(ngram)}")

    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError(f"num_samples must be a positive integer, but got {num_samples}")

    if not isinstance(sequence_length, int) or sequence_length < ngram.n:
        raise ValueError(f"sequence_length must be integer grater or equal than n={ngram.n}, but got {sequence_length}")

    ngram.norm()
    targets = numpy.zeros((num_samples, sequence_length), dtype='int64')

    def gen_sequence(i):
        targets[i, :ngram.n] = numpy.array(ngram.sample())
        for j in range(1, sequence_length - ngram.n + 1):
            x = ngram.subgram(tuple(targets[i, j:j+ngram.n-1])).sample()
            if x is None:
                return False
            targets[i, j+ngram.n-1] = x[0]
        return True

    for i in range(num_samples):
        while not gen_sequence(i):
            pass

    return targets


def extract_ngram(data, n, data_loader=False):
    """Create n-gram dictionary from set of sentences.

    ### Parameters:
    - targets: if `data_loader = False`, then num_samples x sequence_length numpy array. If `data_loader = True`, then it should de data
    loader of labeled sequences extending `torch.utils.data.DataLoader`.
    - n: the `n` from n-gram.
    - data: changes interpretation of `targets` parameter.

    ### Returns:
    - n-gram distribution (`spdg.Ngram`)
    """
    ngram = Ngram(n)

    if data_loader:

        for _, y in data:
            for sample in y:
                ngram[tuple(sample.to('cpu').numpy())] += 1

    else:

        for sentence in data.astype('int64'):
            for i in range(len(sentence) - n + 1):
                ngram[tuple(sentence[i:i+n])] += 1

    return ngram.norm()


class MNISTSequencesDataset(torch.utils.data.Dataset):
    """Create dataset containing sequences of MNIST digits.

    ### Parameters:
    - n: sampled sequence length (returned by __get__, should be the `n` from n-gram).
    - targets: num_samples x sequence_length numpy array of some sequences (sequence_length >= n) of digits 0, 1...9. Those will
    be mapped to MNIST digits.
    - train: if `True` (default), then uses MNIST train images. If 'False', then uses MNIST test images.
    - root: root location, where MNIST data set should be located. If not found, then it will be downloaded to this location.
    - image_choice_policy: either 'replacement', 'shuffle', or 'as-is'. By default 'shuffle'. If 'replacement', then
    images are sampled randomly with replacement. If 'as-is', then they are assigned with order in mnist data set to sequences
    (cyclically if necessary). If 'shuffle', then they are shuffled beforehand.
    - transform: torchvision transform applied to every MNIST image. Default: `Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])`
    """
    def __init__(self, n, targets, train=True, root='./', image_sampling_policy='shuffle', transform=None):

        if image_sampling_policy != 'shuffle' and image_sampling_policy != 'replacement' and image_sampling_policy != 'as-is':
            raise ValueError("Invalid 'image_sampling_policy' parameter. Possible values are 'replacement', 'shuffle' and 'as-is'.")

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)), ])
        else:
            self.transform = transform

        mnist = datasets.MNIST(root, train=train, download=True)

        mnist_size = dict()
        for i in range(10):
            mnist_size[i] = mnist.data.numpy()[mnist.targets.numpy() == i].shape[0]

        self.data = numpy.zeros((targets.shape[0], targets.shape[1], 28, 28), dtype='float32')

        self.n = n
        self.targets = targets.astype('int64')

        if image_sampling_policy == 'replacement':
            for i in range(10):
                if (self.targets == i).any():
                    pos = numpy.random.randint(mnist_size[i], size=(self.targets == i).sum())
                    self.data[self.targets == i, ...] = mnist.data.numpy()[mnist.targets.numpy() == i][pos]

        else:
            indices = dict()

            if image_sampling_policy == 'shuffle':
                for i in range(10):
                    indices[i] = numpy.random.permutation(mnist_size[i])
            else:
                for i in range(10):
                    indices[i] = numpy.arange(mnist_size[i])

            for i in range(10):
                if (self.targets == i).any():
                    size = (self.targets == i).sum()
                    while size > len(indices[i]):
                        indices[i] = numpy.concatenate((indices[i], indices[i]))
                    self.data[self.targets == i, ...] = mnist.data.numpy()[mnist.targets.numpy() == i][indices[i][:size]]

    def __len__(self):
        return self.data.shape[0] * (self.data.shape[1] - self.n + 1)

    def __getitem__(self, index):
        sample = index // (self.data.shape[1] - self.n + 1)
        offset = index % (self.data.shape[1] - self.n + 1)

        res = self.transform(self.data[sample][offset])
        for i in range(1, self.n):
            res = torch.cat((res, self.transform(self.data[sample][offset + i])), 0)

        return res, self.targets[sample][offset:offset+self.n]

    def get(self, index):
        res = self.transform(self.data[index][0])

        for i in range(1, self.data.shape[1]):
            res = torch.cat((res, self.transform(self.data[index][i])), 0)

        return res, self.targets[index]


class MNISTSequencesLoader(torch.utils.data.DataLoader):
    """Wrapper class for torch DataLoader around MNISTSequencesDataset.

    ### Parameters:
    - n: sampled sequence length (returned by __get__, should be the `n` from n-gram).
    - targets: num_samples x sequence_length numpy array of some sequences (sequence_length >= n) of digits 0, 1...9. Those will
    be mapped to MNIST digits.
    - train: if `True` (default), then uses MNIST train images. If `False`, then uses MNIST test images.
    - root: root location, where MNIST data set should be located. If not found, then it will be downloaded to this location.
    - image_choice_policy: either 'replacement', 'shuffle', or 'as-is'. By default 'shuffle'.
        - 'replacement': images are sampled randomly with replacement;
        - 'as-is': images are assigned with order in mnist data set to sequences (cyclically if necessary);
        - 'shuffle': it's like 'as-is', but they are shuffled beforehand.
    - transform: torchvision transform applied to every MNIST image. Default: `Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])`
    - shuffle, batch_size, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn,
    multiprocessing_context: parameters from
    [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html)
    """
    def __init__(self, n, targets, batch_size=1, shuffle=False, train=True, root='./', image_sampling_policy='shuffle', transform=None,
                 sampler=None, batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):

        dataset = MNISTSequencesDataset(n, targets, train, root, image_sampling_policy, transform)

        super(MNISTSequencesLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn,
                                                   pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context)
