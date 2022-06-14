import torch

class Args:
    TAGS = {'': 0, 'B': 1, 'I': 2, 'O': 3}
    model_name = "bert"
    model_path = "model_hub/voidful-albert-chinese-tiny/"
    if model_name == "bert":
        max_seq_len = 48  # title的最大长度
    else:
        max_seq_len = 64
    tag_num = 1   # tag的数目，这里只使用品牌属性，因此数目是1
    tag_max_len = 4  # tag的最大长度
    pickle_path = 'data/品牌.pkl'  # 先再preprocess.py里面生成指定的文件，再在这里指定
    batch_size = 128
    num_workers = 2
    embedding_dim = 312
    hidden_dim = 312
    tagset_size = 4  # 0:pad,1:B,2:I,3:O
    do_train = False
    do_test = False
    do_predict = True
    if do_train:
      load_model_path = None
    else:
      load_model_path = "./checkpoints/BertForAE_0614_06:20:14.pth"  # 训练完成后在这里设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = 2e-5
    crf_lr = 2e-3
    other_lr = 2e-4
    weight_decay = 0.001
    adam_epsilon = 1e-9
    print_freq = 100
    dropout = 0.1
    seed = 123
    max_epoch = 10