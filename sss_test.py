import re
from eunjeon import Mecab
mecab = Mecab()
hangul = re.compile('[^0-9a-zA-Z가-힣\s]')


def sss(text):
    text = re.sub(r" {2,}", " ", hangul.sub(' ', text))
    end_char = ['요', '다', '죠']
    avoid_char = ['보다', '하려다', '하다', '려다']
    special_char = ['느림']
    new_sentences = []
    ts = text.split()
    start, end, flag = 0, 0, 0
    for i in range(len(ts)):
        if (len(ts[i]) >= 2 and ts[i][-1] in end_char and ts[i][-2:] not in avoid_char) \
                or ('ETN' == mecab.pos(ts[i])[-1][1].split('+')[-1]) or (ts[i][-2:] in special_char):  # and mecab.pos(ts[i])[-1][1] != 'NNG'
            end = i
            new_sentences.append(' '.join(ts[start:end + 1]).strip())
            start = end + 1
            flag = 1
        else:
            if i == len(ts) - 1:
                new_sentences.append(' '.join(ts[start:]).strip())
    if not flag and len(new_sentences) == 0:
        new_sentences.append(text)
    return new_sentences

print(sss('원단이 두툼한것치곤 가격은 저렴하게 산듯하고 이계절입긴 괜츈 엠 X쥬 보통체격잘맞네요'))
print(sss('잘 입긴 함'))
