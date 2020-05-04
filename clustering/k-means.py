import numpy
import gensim
import nltk.cluster.util
from nltk.cluster.kmeans import KMeansClusterer


if __name__ == '__main__':
    model = gensim.models.Word2Vec.load('../word2vec/word2vec_model')
    f = open('../word2vec/vocabulary.txt')
    phrases = f.read().strip().split(' ')

    arr = numpy.empty((0, 100), float)
    embedded_phrases = []
    f = open('../AutoPhrase/models/cate/AutoPhrase_multi-words.txt')
    multi_words = f.read().strip().split('\n')
    for i in multi_words:
        ii = i.split('\t')
        score = float(ii[0])
        phrase = ii[1].replace(' ', '_')
        if score < 0.5:
            break
        try:
            arr = numpy.append(arr, numpy.reshape(model.wv.word_vec(phrase), (1, 100)), axis=0)
        except KeyError:
            pass
        else:
            embedded_phrases.append(phrase)

    f = open('../AutoPhrase/models/cate/AutoPhrase_single-word.txt')
    single_word = f.read().strip().split('\n')
    for i in single_word:
        ii = i.split('\t')
        score = float(ii[0])
        phrase = ii[1]
        if score < 0.7:
            break
        try:
            arr = numpy.append(arr, numpy.reshape(model.wv.word_vec(phrase), (1, 100)), axis=0)
        except KeyError:
            pass
        else:
            embedded_phrases.append(phrase)

    print('number of sample points:', len(embedded_phrases))

    kmeans = KMeansClusterer(6, nltk.cluster.util.cosine_distance)
    clusters = kmeans.cluster(arr, assign_clusters=True)
    centers = kmeans.means()

    result = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    for i in range(len(clusters)):
        result[clusters[i]].append([nltk.cluster.util.cosine_distance(centers[clusters[i]], arr[i]),
                                    embedded_phrases[i]])
    for k in result:
        sorted_result = sorted(result[k], reverse=True)
        final_result = '\n'.join(['%.10f' % x[0] + '\t' + x[1] for x in sorted_result])
        f = open('cluster' + str(k) + '.txt', 'w+')
        f.write(final_result)
