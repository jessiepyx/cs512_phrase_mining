def build_vocab(file):
    vocab = []
    out = open('vocabulary.txt', 'w+')
    lines = file.readlines()
    for line in lines:
        vocab.append(line.strip().split('\t')[1].replace(' ', '_'))
    out.write(' '.join(vocab))


if __name__ == "__main__":
    f = open('../AutoPhrase/models/cate/AutoPhrase.txt')
    build_vocab(f)
