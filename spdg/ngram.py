from collections import defaultdict

import numpy

from .errors import ParameterError


class Ngram(defaultdict):
    """Class implementation of n-gram probabilities in form of dictionary"""

    def __init__(self, n):
        """Create empty n-gram"""
        super(Ngram, self).__init__(int)

        if not isinstance(n, int) or n < 1:
            raise ParameterError(f"n must be positive integer, but got {n}")

        self.n = n
        self.subgrams = dict()

    def sum(self):
        return sum([self[x] for x in self])

    def norm(self):
        """Normalize n-gram - required to call before sampling"""
        _s = self.sum()
        for _x in self:
            self[_x] /= _s
        return self

    def sample(self):
        """Sample random entry with corresponding probabilities."""
        x = numpy.random.rand()
        for idx in self:
            x -= self[idx]
            if x < 0:
                return idx
        return None

    def parameters(self):
        for idx in self:
            yield self[idx]

    def __str__(self):
        obj = ''
        for idx in self:
            obj = '{}{}: {:.2f}%\n'.format(obj, idx, 100. * self[idx])
        return obj.rstrip()

    def size(self):
        return len(self)

    def subgram(self, v, cache=True):
        """Create subgram (unigram) for given entry.

        Parameters:

        - v (tuple of ints): prefix of length n-1 of some ngram entries (eg. '(1, 2)' for 3-gram)
        - cache (bool, optional): if set to `true`, then returns cached last result.

        Returns:
        - ngram: possible continuations of given prefix in form of unigram.
        """

        if len(v) != self.n - 1:
            raise NotImplementedError("""ngram.subgram does not handle indices resulting with k-grams for k > 1""")

        if cache and v in self.subgrams.keys():
            return self.subgrams[v]

        self.subgrams[v] = Ngram(1)

        for idx in self:
            ok = True
            for i in range(len(v)):
                if v[i] != idx[i]:
                    ok = False
            if ok:
                self.subgrams[v][idx[len(v)]] = self[idx]

        return self.subgrams[v].norm()


def from_data(data, n):
    """Create n-gram dictionary from set of sentences.

    Parameters:

    - data: num_samples x sample_length numpy array of ints
    - n: as in n-gram

    Returns: resulting n-gram distribution.
    """
    ngram = Ngram(n)

    for sequence in data.astype('int64'):
        for i in range(len(sequence) - n + 1):
            ngram[tuple(sequence[i:i+n])] += 1

    return ngram.norm()


def from_data_loader(data_loader, n):
    """Create ngram from data loader of labeled sequences.

    Parameters:
    - data_loader: pytorch data loader of labeled sequences
    - n: as in n-gram

    Returns: n-gram distribution
    """
    ngram = Ngram(n)

    for _, y in data_loader:
        for sample in y:
            ngram[tuple(sample.to('cpu').numpy())] += 1

    return ngram.norm()


def random(n, entries, k=10, min_var=0):
    """Randomly creates n-gram containing tuples (a_1, ... a_n) where a_i ∈ {0, 1, ... k - 1}.

    Parameters:
    - n: as in n-gram
    - entries: expected number of entries in resulting n-gram
    - k: number of symbols used in n-gram creation
    - min_var: minium variance of n-gram probabilities

    Returns: n-gram
    """
    if not isinstance(k, int) or k < 1:
        raise ParameterError(f"'k' must be positive integer, but got {k}")

    if not isinstance(entries, int) or entries < 0:
        raise ParameterError(f"'entries' must be non negative integer, but got {entries}")

    if entries > k ** n:
        raise ParameterError(f"Cannot create n-gram with given constraints. Maximum number of entries in {n}-gram with {k} \
                available symbols is {k ** n}, but got requested {entries}.")

    ngram = Ngram(n)

    while ngram.size() < entries:
        ngram[tuple(numpy.random.randint(0, k, n))] = numpy.random.random()

    unique = set()

    for idx in ngram:
        for i in idx:
            unique.add(i)

    if len(unique) != k:
        return random(n, entries, k, min_var)

    ngram.norm()

    if min_var > 0:
        mu = sum(ngram.values()) / entries
        var = sum([(x - mu) ** 2 for x in ngram.values()]) / entries

        if var < min_var:
            return random(n, entries, k, min_var)

    return ngram
