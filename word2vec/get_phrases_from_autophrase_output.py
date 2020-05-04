import re


def preprocess(file):
    content = file.read()
    regex = re.compile('<phrase>(.*?)</phrase>')

    def repl(match):
        return match.group(1).replace(' ', '_')

    result = regex.sub(repl, content)
    out = open('phrases.txt', 'w+')
    out.write(result)

    phra = len(regex.findall(content))
    sent = len(content.strip().split('\n'))
    print('number of qualified phrases:', phra)
    print('number of sentences:', sent)
    print('average number of phrases per sentence:', float(phra) / sent)


if __name__ == "__main__":
    f = open('../AutoPhrase/models/cate/segmentation.txt')
    preprocess(f)
