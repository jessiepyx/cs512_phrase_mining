import argparse
import numpy
from get_phrases_from_autophrase_output import preprocess
from get_vocabulary_from_autophrase_output import build_vocab
import gensim


def word2vec(content):
    words = [content.split(' ')]
    model = gensim.models.Word2Vec(words, sg=1, size=100, window=5, negative=0, hs=1, sample=1e-3, workers=12,
                                   min_count=1)
    model.save('word2vec_model')
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Step2 Phrase Embedding')

    parser.add_argument('-p', '--preprocess', action='store_true', default=False,
                        help='Flag for preprocess: "<phrase>hello world</phrase>" -> "hello_world". Default: False')
    parser.add_argument('-v', '--vocabulary', action='store_true', default=False,
                        help='Flag for building vocabulary from AutoPhrase.txt. Default: False')
    parser.add_argument('-i', '--input', type=str, default='phrases.txt',
                        help='Input file path. Default: ./phrases.txt')
    parser.add_argument('-n', '--num', type=int, default=100,
                        help='Number of top phrases included in the result.')

    args = parser.parse_args()

    file = open(args.input)
    if args.vocabulary:
        f = open('../AutoPhrase/models/cate/AutoPhrase.txt')
        build_vocab(f)
    if args.preprocess:
        preprocess(file)
        f = open('phrases.txt')
        content = f.read()
    else:
        content = file.read()
    model = word2vec(content)

    file = open('vocabulary.txt')
    vocab = file.read().split(' ')
    result = []
    for i in range(min(len(vocab), args.num)):
        try:
            result.append(vocab[i] + ' ' + numpy.array_str(model[vocab[i]], max_line_width=numpy.inf)[1:-1].strip())
        except:
            print(vocab[i])

    out = open('embedding.txt', 'w+')
    out.write('\n'.join(result))
