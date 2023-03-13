"""bAbI question answering dataset."""

import glob
import itertools
import os
import shutil
import urllib.request
from functools import reduce
from typing import TypedDict, Any, Tuple, List, Dict

import numpy as np
import torch.utils.data


class BABIDataset(torch.utils.data.Dataset):
    """Sentence level bAbI dataset."""

    url = 'https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz'

    def __init__(self, root: str, task: int, train: bool = True, ten_k: bool = True, max_num_sentences: int = None,
                 download: bool = True) -> None:
        super().__init__()
        self.root = root
        self.task = task
        self.train = train
        self.ten_k = ten_k

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')

        if self.task < 1 or self.task > 20:
            raise ValueError('Task {0} not available. Use one of the following: 1,2,...,20.'.format(task))

        self.stories, self.stats = self._parse_stories(max_num_sentences)

        self.data = None

    def __len__(self) -> int:
        if self.data is None:
            raise RuntimeError('Vectorize the stories first by using `vectorize_stories`')

        return len(self.data[0])

    def __getitem__(self, idx: int) -> Tuple[TypedDict('BABIDatasetItem', {'story': Any, 'query': Any, 'answer': Any}),
                                             int]:
        if self.data is None:
            raise RuntimeError('Vectorize the stories first by using `vectorize_stories`')

        story = self.data[0][idx]
        query = self.data[1][idx]
        answer = self.data[2][idx]

        # Time steps of story without padding.
        story_length = np.count_nonzero(story, axis=0)[0]

        return {'story': torch.from_numpy(story), 'query': torch.from_numpy(query), 'answer': answer}, story_length

    @property
    def folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    @staticmethod
    def build_vocab(stories: List, max_num_sentences: int, add_time_words: bool) -> Tuple[List, int]:
        vocab = sorted(reduce(lambda x, y: x | y, (set(list(itertools.chain.from_iterable(s)) + q + a)
                                                   for s, q, a in stories)))
        vocab.insert(0, 'NIL')
        vocab_size = len(vocab)

        if add_time_words:
            vocab += ['TIME{}'.format(i + 1) for i in range(max_num_sentences)]

        return vocab, vocab_size

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.folder, 'tasks_1-20_v1-2'))

    def _parse_stories(self, max_sentences: int) -> Tuple[List, Dict]:
        path = os.path.join(self.folder, 'tasks_1-20_v1-2/en-10k' if self.ten_k else 'tasks_1-20_v1-2/en')
        filename = 'qa{0}_*_{1}*'.format(self.task, 'train' if self.train else 'test')
        filepath = glob.glob(os.path.join(path, filename))[0]

        story_sizes = []
        with open(filepath) as f:
            stories, story = [], []
            for line in f.readlines():
                line = line.lower()  # make lowercase
                tid, text = line.rstrip('\n').split(' ', 1)
                if tid == '1':
                    story = []
                # Sentence
                if text.endswith('.'):
                    story.append(text[:-1].split(' '))
                # Question
                else:
                    # Remove any leading or trailing whitespace after splitting.
                    query, answer, supporting = (x.strip() for x in text.split('\t'))
                    sub_story = [x for x in story if x]
                    if max_sentences and len(sub_story) > max_sentences:
                        continue
                    stories.append((sub_story, query[:-1].split(' '), [answer]))  # remove '?'
                    story.append('')

                    story_sizes.append(len(sub_story))

        if not stories:
            raise RuntimeError('No stories parsed. Try increasing max_sentences.')

        max_num_words_story = max(map(len, itertools.chain.from_iterable(s for s, _, _ in stories)))
        max_num_words_query = max(map(len, itertools.chain(q for _, q, _ in stories)))

        stats = {'max_num_sentences': max(story_sizes),
                 'mean_num_sentences': sum(story_sizes) // len(story_sizes),
                 'max_num_words': max(max_num_words_story, max_num_words_query),
                 'max_num_words_story': max_num_words_story,
                 'max_num_words_query': max_num_words_query}

        return stories, stats

    def download(self) -> None:
        if self._check_exists():
            return

        os.makedirs(self.folder, exist_ok=True)
        print('Downloading {0}...'.format(self.url))
        filename = os.path.basename(self.url)
        filepath = os.path.join(self.folder, filename)
        urllib.request.urlretrieve(self.url, filepath)
        shutil.unpack_archive(filepath, self.folder)
        os.remove(filepath)
        print('Done!')

    def vectorize_stories(self, vocab: List, sentence_size: int, add_time_words: bool, padding: str) -> None:
        if padding not in ['pre', 'post']:
            raise ValueError('Padding {0} not available. Use either pre or post'.format(padding))

        word2index = {w: i for i, w in enumerate(vocab)}

        s, q, a = [], [], []
        for story, query, answer in self.stories:
            vectorized_story = []
            for i, sentence in enumerate(story, 1):
                # Pad to pad_len, i.e., add nil words, and add story.
                ls = max(0, sentence_size - len(sentence))
                vectorized_story.append([word2index[w] for w in sentence] + [0] * ls)

            if add_time_words:
                # Make the last word of each sentence the time 'word'.
                for i, vectorized_sentence in enumerate(reversed(vectorized_story), 1):
                    vectorized_sentence[-1] = word2index['TIME{}'.format(i)]

            # Pad stories to max_num_sentences (i.e., add empty stories).
            ls = max(0, self.stats['max_num_sentences'] - len(vectorized_story))
            for _ in range(ls):
                if padding == 'pre':
                    vectorized_story.insert(0, [0] * sentence_size)
                else:
                    vectorized_story.append([0] * sentence_size)

            # Pad queries to pad_len (i.e., add nil words).
            lq = max(0, sentence_size - len(query))
            vectorized_query = [word2index[w] for w in query] + [0] * lq

            vectorized_answer = word2index[answer[0]]

            s.append(vectorized_story)
            q.append(vectorized_query)
            a.append(vectorized_answer)

        self.data = (np.array(s).astype(np.int64), np.array(q).astype(np.int64), np.array(a).astype(np.int64))
