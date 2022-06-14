import os
from transformers import AutoModel, BertTokenizer
from tqdm import tqdm
import pandas as pd
import pickle
import random
import numpy as np
from collections import Counter
from config import Args

args = Args()

def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def bert4token(tokenizer, title, attribute, value):
    title = tokenizer.tokenize(title)
    attribute = tokenizer.tokenize(attribute)
    
    tag = ['O'] * len(title)
    if value:
      for i in range(0, len(title) - len(value)):
          if title[i:i + len(value)] == value:
              for j in range(len(value)):
                  if j == 0:
                      tag[i + j] = 'B'
                  else:
                      tag[i + j] = 'I'
      value = tokenizer.tokenize(value)
      value_id = tokenizer.convert_tokens_to_ids(value)
    else:
      value_id = []
    title = ['[CLS]'] + title + ['[SEP]']
    attribute = ['[CLS]'] + attribute + ['[SEP]']
    title_id = tokenizer.convert_tokens_to_ids(title)
    attribute_id = tokenizer.convert_tokens_to_ids(attribute)
    # 加上CLS和SEP
    tag = ['O'] + tag + ['O']
    tag_id = [args.TAGS[_] for _ in tag]
    return title_id, attribute_id, value_id, tag_id

max_len = args.max_seq_len
tag_max_len = args.tag_max_len


def X_padding(ids):
    if len(ids) >= max_len:
        return ids[:max_len]
    ids.extend([0] * (max_len - len(ids)))
    return ids


def tag_padding(ids):
    if len(ids) >= tag_max_len:
        return ids[:tag_max_len]
    ids.extend([0] * (tag_max_len - len(ids)))
    return ids

def bert_texts(queries, texts, args):
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    titles = []
    attributes = []
    values = []
    tags = []
    for query in queries:
      for text in texts:
        title, attribute, value, tag = bert4token(tokenizer, text, query, "")
        titles.append(title)
        attributes.append(attribute)
        tags.append(tag)
    if titles:
      df = pd.DataFrame({'titles': titles, 'attributes': attributes, 'tags': tags},
                                  index=range(len(titles)))

      df['x'] = df['titles'].apply(X_padding)
      df['y'] = df['tags'].apply(X_padding)
      df['att'] = df['attributes'].apply(tag_padding)
      x = np.asarray(list(df['x'].values))
      att = np.asarray(list(df['att'].values))
      y = np.asarray(list(df['y'].values))
      return x, att, y
    else:
      assert "请输入titles"

def rawdata2pkl4bert(path, att_list):
    if len(att_list[0]) == 2:
        att_list = [i[0] for i in att_list]
    tokenizer = BertTokenizer.from_pretrained('model_hub/voidful-albert-chinese-tiny')
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for att_name in tqdm(att_list):
            print('#' * 20 + att_name + '#' * 20)
            titles = []
            attributes = []
            values = []
            tags = []
            for index, line in enumerate(tqdm(lines, ncols=100)):
                line = line.strip('\n')
                if line:
                    title, attribute, value = line.split('<$$$>')
                    if attribute in [att_name] and value in title:  # and _is_chinese_char(ord(value[0])):
                        title, attribute, value, tag = bert4token(tokenizer, title, attribute, value)
                        titles.append(title)
                        attributes.append(attribute)
                        values.append(value)
                        tags.append(tag)

            if titles:
                print([tokenizer.convert_ids_to_tokens(i) for i in titles[:3]])
                print([[id2tags[j] for j in i] for i in tags[:3]])
                print([tokenizer.convert_ids_to_tokens(i) for i in attributes[:3]])
                print([tokenizer.convert_ids_to_tokens(i) for i in values[:3]])
                df = pd.DataFrame({'titles': titles, 'attributes': attributes, 'values': values, 'tags': tags},
                                  index=range(len(titles)))
                print(df.shape)

                df['x'] = df['titles'].apply(X_padding)
                df['y'] = df['tags'].apply(X_padding)
                df['att'] = df['attributes'].apply(tag_padding)

                index = list(range(len(titles)))
                random.shuffle(index)
                train_index = index[:int(0.85 * len(index))]
                valid_index = index[int(0.85 * len(index)):int(0.95 * len(index))]
                test_index = index[int(0.95 * len(index)):]

                train = df.loc[train_index, :]
                valid = df.loc[valid_index, :]
                test = df.loc[test_index, :]

                train_x = np.asarray(list(train['x'].values))
                train_att = np.asarray(list(train['att'].values))
                train_y = np.asarray(list(train['y'].values))

                valid_x = np.asarray(list(valid['x'].values))
                valid_att = np.asarray(list(valid['att'].values))
                valid_y = np.asarray(list(valid['y'].values))

                test_x = np.asarray(list(test['x'].values))
                test_att = np.asarray(list(test['att'].values))
                test_value = np.asarray(list(test['values'].values))
                test_y = np.asarray(list(test['y'].values))

                att_name = att_name.replace('/', '_')
                with open('data/{}.pkl'.format(att_name), 'wb') as outp:
                    # with open('../data/top105_att.pkl', 'wb') as outp:
                    pickle.dump(train_x, outp)
                    pickle.dump(train_att, outp)
                    pickle.dump(train_y, outp)
                    pickle.dump(valid_x, outp)
                    pickle.dump(valid_att, outp)
                    pickle.dump(valid_y, outp)
                    pickle.dump(test_x, outp)
                    pickle.dump(test_att, outp)
                    pickle.dump(test_value, outp)
                    pickle.dump(test_y, outp)


def get_attributes(path, k=10):
    atts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                title, attribute, value = line.split('<$$$>')
                atts.append(attribute)
    counter_atts = Counter(atts)
    total = len(counter_atts)
    return total, [(item[0], item[1]) for item in counter_atts.most_common(k)]


if __name__ == '__main__':
    """
        该github主提供的数据每一条数据就只有一个属性值，
        因此不可能将其直接转换为序列标注来做，考虑的办法
        是每次可以采取一个属性训练，并对接下来的其它没有
        该属性的句子进行预测补全标签信息，最后再融合起来
        训练，这里先采取品牌进行训练。
    """

    TAGS = args.TAGS
    # TAGS = {'': 0, 'B': 1, 'I': 2, 'O': 3}
    id2tags = {v: k for k, v in TAGS.items()}
    """
    path = 'data/raw.txt'
    topk = 2
    total, att_list = get_attributes(path, topk)
    print("总共有属性：", total)
    if topk is None:
        print("选取的属性：", total)
    else:
        print("选取的属性：", topk)
    print("前10属性是：")
    print(att_list)
    rawdata2pkl4bert(path, att_list)
    # rawdata2pkl4nobert(path)
    """

    queries = ['品牌']
    texts = [
      "y日着原创设计2019夏季新款 蓝色基础百搭t恤女",
      "美国采购bugaboo bee3/bee5婴儿推车自立式支架  推车配件小黑尾",
      "孕之彩2019夏季新款短袖孕妇裙ywq292941休闲条纹针织t恤连衣裙",
      "丽婴房 彼得兔夏装女童时尚薄款清凉连衣裙子宝宝裙子",
      "背带牛仔裤阔腿裤印花拼接文艺森女复古范大码长裤2019春季新款",
      "南极人男士睡衣夏季纯棉短袖长裤薄款宽松加肥加大码家居服男套装",

    ]

    x, att, y = bert_texts(queries, texts, args)
    print(x)
    print(att)
    print(y)
