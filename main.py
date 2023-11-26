import time

import numpy as np
import torch
from model import SASRec
from utils import *
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# 用于绘制图像查看迭代过程中的NDCG和HR
def plot_and_save(x, y, title, filename):
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()


# 自定义数据集 用于dataloader分批次读取数据
class TrainDataset(Dataset):
    def __init__(self, train_data, sequence_length, usernum, itemnum):
        self.train_data = train_data
        self.sequence_length = sequence_length
        self.usernum = usernum
        self.itemnum = itemnum

    def __len__(self):
        return len(self.train_data)

    def random_neg(self, l, r, ts):
        t = np.random.randint(l, r)
        while t in ts:
            t = np.random.randint(l, r)
        return t

    def __getitem__(self, index):
        # 注意这里的getitem是用于dataloader读取数据的，所以这里的index是dataloader传入的，但是这里的index是无用的，因为我们是随机取的数据
        userid = np.random.randint(1, self.usernum + 1)
        while len(self.train_data[userid]) <= 1: userid = np.random.randint(1, self.usernum + 1)

        seq = np.zeros([self.sequence_length], dtype=np.int32)
        pos = np.zeros([self.sequence_length], dtype=np.int32)
        neg = np.zeros([self.sequence_length], dtype=np.int32)

        # 根据论文中的描述，我们需要将序列中的物品分为正样本和负样本，正样本就是序列中的物品，负样本就是随机抽取的物品
        # 而我们的序列 你可以发现 seq和pos是不一样的 seq是序列中的物品，而pos是seq序列向后移动一位的结果，这样就可以得到正样本
        # 也就是说 pos是真正训练集序列的后50个物品，而seq要整体向左移动一位 因为要以seq预测pos
        # 根据原论文  假设我们取的序列长度是50，而真实序列长度不够50  则记得再前面补0 即填充项，然后在后续的处理中过滤掉填充项
        offset_item = self.train_data[userid][-1]
        idx = self.sequence_length - 1
        ts = set(self.train_data[userid])
        for i in reversed(self.train_data[userid][:-1]):
            seq[idx] = i
            pos[idx] = offset_item
            if offset_item != 0: neg[idx] = self.random_neg(1, self.itemnum + 1, ts)
            offset_item = i
            idx -= 1
            if idx == -1: break
        return (userid, seq, pos, neg)


if __name__ == '__main__':
    path = "./data/ml-1m.txt"
    sequence_length = 50
    batch_size = 128
    epochs = 200
    hidden_units = 50
    dropout_rate = 0.5
    num_of_blocks = 2
    num_of_heads = 1
    learning_rate = 0.001

    [train_data, valid_data, test_data, user_num, item_num] = load_data(path)
    print("finish loading data...")
    train_dataset = TrainDataset(train_data, sequence_length, user_num, item_num)
    print("finish creating dataset...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    print("start training...")

    model = SASRec(user_num, item_num, hidden_units, dropout_rate, sequence_length, num_of_blocks, num_of_heads)
    # 初始化权重
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass  # just ignore those failed init layers
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98))

    model.train()  # 训练模式
    t0 = time.time()

    epoch_list = []
    NDCG_list = []
    HR_list = []

    num = 0
    f = open(f"./saved_results/result{num}.txt", "w")
    for epoch in range(1, epochs + 1):
        print(epoch)
        for userid, seq, pos, neg in train_dataloader:
            # 分别根据数据对取出来的正样本和负样本进行预测
            pos_predictions, neg_predictions = model(seq, pos, neg)
            pos_labels = torch.ones_like(pos_predictions)
            neg_labels = torch.zeros_like(neg_predictions)

            adam_optimizer.zero_grad()
            real_labels_index = np.where(pos != 0) # 过滤掉填充项
            loss = bce_criterion(pos_predictions[real_labels_index], pos_labels[real_labels_index])
            loss += bce_criterion(neg_predictions[real_labels_index], neg_labels[real_labels_index])
            loss.backward()
            adam_optimizer.step()

        if epoch % 10 == 0:
            model.eval()
            epoch_list.append(epoch)
            NDCG, HR = evaluate(model, [train_data, valid_data, test_data, user_num, item_num], sequence_length)
            NDCG_list.append(NDCG)
            HR_list.append(HR)
            print("epoch: ", epoch, " NDCG: ", NDCG, " HR: ", HR)
            f.write(f"epoch: {epoch}, NDCG: {NDCG},  HR: {HR},  loss: {loss.item()}\n")
            f.flush()
            NDCG, HR = evaluate(model, [train_data, valid_data, test_data, user_num, item_num], sequence_length,True)
            print("(validate)epoch: ", epoch, " NDCG: ", NDCG, " HR: ", HR)
            model.train()
    f.close()

    plot_and_save(epoch_list, NDCG_list, "NDCG", "./saved_results/NDCG.png")
    plot_and_save(epoch_list, HR_list, "HR", "./saved_results/HR.png")
    print("Finished....")
    plt.close()
