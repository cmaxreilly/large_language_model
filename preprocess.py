import torchtext
from torchtext.data.utils import get_tokenizer
from collections import Counter

tokenizer = get_tokenizer('basic_english')

def build_vocab(data, vocab_size):
    counter = Counter()
    for line in data:
        counter.update(tokenizer(line))
    return torchtext.vocab.Vocab(counter, min_freq=1, max_size=vocab_size)

def preprocess(filename, vocab_size):
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
    vocab = build_vocab(data, vocab_size)
    tokenized_data = []
    for line in data:
        tokens = tokenizer(line.strip())
        token_ids = [vocab.stoi[token] for token in tokens]
        tokenized_data.append(token_ids)
    return tokenized_data, vocab
