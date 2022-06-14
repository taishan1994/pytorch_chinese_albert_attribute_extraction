import os
import sys
from time import strftime, localtime
from collections import Counter
from transformers import BertTokenizer, AdamW
import random
import numpy as np
import torch
from preprocess import bert_texts
from model import BertForAE
from data_loader import get_dataloader, MyDataset
# from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.metrics import f1_score, accuracy_score, classification_report
from seqeval.metrics.sequence_labeling import get_entities
get_entities
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def get_attributes(path='./data/raw.txt'):
    atts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            if line:
                title, attribute, value = line.split('<$$$>')
                atts.append(attribute)
    return [item[0] for item in Counter(atts).most_common()]


def train(args):
    print('开始训练。。。')
    log_file = '{}.log'.format(args.model_name, strftime("%y%m%d-%H%M", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    att_list = get_attributes()

    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    tags2id = args.TAGS
    id2tags = {v: k for k, v in tags2id.items()}

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # step1: configure model
    model = BertForAE(args)
    if args.load_model_path:
        model.load(args.load_model_path)
    model.to(args.device)

    # step2: data
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args)

    # step3: criterion and argsimizer
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        print(name)
        space = name.split('.')
        # print(name)
        if space[0] == 'bert':
            bert_param_optimizer.append((name, para))
        elif space[0] == 'crf':
            crf_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.lr},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.lr},

        # crf模块
        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.crf_lr},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.crf_lr},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.other_lr},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.other_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)

    # step4 train
    for epoch in range(args.max_epoch):
        model.train()
        for ii, batch in enumerate(tqdm(train_dataloader, ncols=100)):
            # train model
            optimizer.zero_grad()
            x = batch['x'].to(args.device)
            y = batch['y'].to(args.device)
            att = batch['att'].to(args.device)
            inputs = [x, att, y]
            loss = model.log_likelihood(inputs)
            loss.backward()
            # CRF
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3)
            optimizer.step()
            if ii % args.print_freq == 0:
                print('\n')
                print('epoch:%04d,------------loss:%f' % (epoch, loss.item()))

    model.save()

    preds, labels = [], []
    for index, batch in enumerate(valid_dataloader):
        model.eval()
        x = batch['x'].to(args.device)
        y = batch['y'].to(args.device)
        att = batch['att'].to(args.device)
        inputs = [x, att, y]
        predict = model(inputs)

        if index % 5 == 0:
            print(tokenizer.convert_ids_to_tokens([i.item() for i in x[0].cpu() if i.item() > 0]))
            length = [id2tags[i.item()] for i in y[0].cpu() if i.item() > 0]
            print(length)
            print([id2tags[i] for i in predict[0][:len(length)]])

        # 统计非0的，也就是真实标签的长度
        leng = []
        for i in y.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j.item())
            leng.append(tmp)

        for index, i in enumerate(predict):
            preds.append([id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]])
            # preds += i[:len(leng[index])]

        for index, i in enumerate(y.tolist()):
            labels.append([id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]])
            # labels += i[:len(leng[index])]
    # precision = precision_score(labels, preds, average='macro')
    # recall = recall_score(labels, preds, average='macro')
    # f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print(report)
    logger.info(report)

def test(args):
    print("开始测试。。。")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    tags2id = args.TAGS
    id2tags = {v: k for k, v in tags2id.items()}

    # step1: configure model
    model = BertForAE(args)
    if args.load_model_path:
        model.load(args.load_model_path)
    model.to(args.device)
    model.eval()
    # step2: data
    train_dataloader, valid_dataloader, test_dataloader = get_dataloader(args)

    # step4 test
    preds, labels = [], []
    for index, batch in enumerate(valid_dataloader):
        model.eval()
        x = batch['x'].to(args.device)
        y = batch['y'].to(args.device)
        att = batch['att'].to(args.device)
        inputs = [x, att, y]
        predict = model(inputs)

        if index % 5 == 0:
            print(tokenizer.convert_ids_to_tokens([i.item() for i in x[0].cpu() if i.item() > 0]))
            length = [id2tags[i.item()] for i in y[0].cpu() if i.item() > 0]
            print(length)
            print([id2tags[i] for i in predict[0][:len(length)]])

        # 统计非0的，也就是真实标签的长度
        leng = []
        for i in y.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j.item())
            leng.append(tmp)

        for index, i in enumerate(predict):
            preds.append([id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]])
            # preds += i[:len(leng[index])]

        for index, i in enumerate(y.tolist()):
            labels.append([id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]])
            # labels += i[:len(leng[index])]
    # precision = precision_score(labels, preds, average='macro')
    # recall = recall_score(labels, preds, average='macro')
    # f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print(report)


def predict(args, queries, texts):
    print("开始预测。。。")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    tags2id = args.TAGS
    id2tags = {v: k for k, v in tags2id.items()}

    # step1: configure model
    model = BertForAE(args)
    if args.load_model_path:
        model.load(args.load_model_path)
    model.to(args.device)
    model.eval()
    # step2: data
    x, att, y = bert_texts(queries, texts, args)
    dataset = MyDataset(x, y, att)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers)

    # 统计非0的，也就是真实标签的长度
    def get_leng(y):
      leng = []
      for i in y.cpu():
          tmp = []
          for j in i:
              if j.item() > 0:
                  tmp.append(j.item())
          leng.append(tmp)
      return leng

    for index, batch in enumerate(data_loader):
        model.eval()
        x = batch['x'].to(args.device)
        y = batch['y'].to(args.device)
        att = batch['att'].to(args.device)
        inputs = [x, att, y]
        predict = model(inputs)

        leng = get_leng(y)
        att_leng = get_leng(att)

        for index, i in enumerate(predict):
          pred = [id2tags[k] if k > 0 else id2tags[3] for k in i[:len(leng[index])]]
          tokens = tokenizer.convert_ids_to_tokens(x[index][:len(leng[index])])
          text = tokenizer.convert_tokens_to_string(tokens[1:-1])
          att_tokens = tokenizer.convert_ids_to_tokens(att[index][:len(att_leng[index])])
          att_text = tokenizer.convert_tokens_to_string(att_tokens[1:-1])
          print("text:", text.split())
          entities = get_entities(pred)
          print("entities:")
          for entity in entities:
            print([att_text, tokenizer.convert_tokens_to_string(tokens[entity[1]:entity[2]+1])])




def main(args, queries=None, texts=None):
  if args.do_train:
      train(args)
  elif args.do_test:
      test(args)
  elif args.do_predict:
      predict(args, queries, texts)
  else:
      assert "请输入正确的模式"

if __name__ == '__main__':
    from config import Args

    args = Args()
    queries = ['品牌']
    texts = [
      "y日着原创设计2019夏季新款 蓝色基础百搭t恤女",
      "美国采购bugaboo bee3/bee5婴儿推车自立式支架  推车配件小黑尾",
      "孕之彩2019夏季新款短袖孕妇裙ywq292941休闲条纹针织t恤连衣裙",
      "丽婴房 彼得兔夏装女童时尚薄款清凉连衣裙子宝宝裙子",
      "背带牛仔裤阔腿裤印花拼接文艺森女复古范大码长裤2019春季新款",
      "南极人男士睡衣夏季纯棉短袖长裤薄款宽松加肥加大码家居服男套装",

    ]
    main(args, queries, texts)
