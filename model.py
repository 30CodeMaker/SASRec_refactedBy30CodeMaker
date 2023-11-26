import time

import numpy as np
import torch
import torch.nn as nn


# SASRec network————FFN
class FFN(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(FFN, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, input_of_Attention):
        outputs = self.conv2(self.dropout(self.relu(self.conv1(input_of_Attention.transpose(-1, -2)))))
        outputs = outputs.transpose(-1, -2)
        return outputs


class SASRec(nn.Module):
    def __init__(self, user_num, item_num, hidden_units, dropout_rate, sequence_length, num_of_blocks, num_of_heads):
        super(SASRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.item2emb = nn.Embedding(self.item_num + 1, hidden_units, padding_idx=0)
        self.position2emb = nn.Embedding(sequence_length, hidden_units)
        self.sequence_length = sequence_length

        # 根据原论文构建的block 由于block的数量是可变的，所以这里使用了ModuleList
        # 以下分别是多头注意力机制和FFN的layerNorm、dropout和layer
        self.layerNorm_of_multi_head_attention_layers = nn.ModuleList()
        self.multi_head_attention_layers = nn.ModuleList()

        self.layerNorm_of_FFN_layers = nn.ModuleList()
        self.FFN_layers = nn.ModuleList()

        self.drop = nn.Dropout(dropout_rate)
        # 根据块的数量构建block
        for _ in range(num_of_blocks):
            self.layerNorm_of_multi_head_attention_layers.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.multi_head_attention_layers.append(nn.MultiheadAttention(hidden_units, num_of_heads, dropout_rate))

            self.layerNorm_of_FFN_layers.append(nn.LayerNorm(hidden_units, eps=1e-8))
            self.FFN_layers.append(FFN(hidden_units, dropout_rate))

    def sasRecBlock(self, i, inputs, attention_mask):
        # 根据原论文构建sasRecBlock
        inputs = torch.transpose(inputs, 0, 1)
        # 先进行layerNorm然后再进行multi_head_attention多头注意力机制最后再进行dropout
        inputs_norm = self.layerNorm_of_multi_head_attention_layers[i](inputs)
        attention_output, _ = self.multi_head_attention_layers[i](inputs_norm, inputs, inputs,
                                                                  attn_mask=attention_mask)
        attention_output = self.drop(attention_output)
        # residual connection 残差连接
        attention_output += inputs

        # 先进行layerNorm然后再进行FFN最后再进行dropout
        inputs_norm = self.layerNorm_of_FFN_layers[i](attention_output)
        outputs = self.FFN_layers[i](inputs_norm)
        outputs = self.drop(outputs)
        # residual connection 残差连接
        outputs += attention_output
        outputs = torch.transpose(outputs, 0, 1)
        return outputs

    #这个方法用于得到结果n个原论文构建的block的输出
    def getF(self,seqs):
        # 取出不同批次中序列中出现的物品的embedding
        seqs_emb = self.item2emb(seqs)
        # 取出不同批次中序列中物品位置的embedding
        positions = np.tile(np.array(range(seqs.shape[1])), [seqs.shape[0], 1])
        # 将物品位置的embedding加到物品的embedding中
        seqs_emb += self.position2emb(torch.LongTensor(positions))

        # 设置去除填充项的mask 用于过滤不存在的序列片段
        time_line_mask = torch.BoolTensor(seqs == 0)

        # 下三角过滤器 但是为什么要取反呢？？？？？ 不是要保留下三角吗，为什么要把下三角删了  查询发现原来是true表示不使用
        # 至于为什么要只保留下三角，是因为要保证序列的顺序，不能让后面的物品出现在前面 即不能以未来预测现在，只能以过去预测未来，更详细的解释可以参考原论文
        attention_mask = ~torch.tril(torch.ones(self.sequence_length, self.sequence_length, dtype=torch.bool))

        for i in range(len(self.multi_head_attention_layers)):
            #过滤不存在的序列片段
            seqs_emb *= ~time_line_mask.unsqueeze(-1)
            #经过block的处理后得到的结果
            seqs_emb = self.sasRecBlock(i, seqs_emb, attention_mask)
        return seqs_emb

    def forward(self, seqs, pos_samples, neg_samples):
        seqs_emb = self.getF(seqs)#得到对应论文中的F块

        pos_samples_emb = self.item2emb(pos_samples)
        neg_samples_emb = self.item2emb(neg_samples)

        #根据原论文的预测公式对正负样本进行预测
        pos_predictions = torch.sum(seqs_emb * pos_samples_emb, -1)
        neg_predictions = torch.sum(seqs_emb * neg_samples_emb, -1)

        return pos_predictions, neg_predictions

    def predict(self, log_seqs, item_indices): # for inference

        log_feats = self.getF(torch.LongTensor(log_seqs)) #取出该序列对应的embedding
        #预测下一个物品只需要取出最后一个序列步的embedding即可，与原论文保持一致
        # 1*50
        final_feat = log_feats[:, -1, :]

        #取出物品的embedding,注意这里应是传入了一个列表，列表中的元素是物品的id，一共有101个物品，其中只有一个物品是正样本，其余都是负样本
        item_embs = self.item2emb(torch.LongTensor(item_indices))
        #这里交换维度，为了后续的矩阵乘法
        item_embs = item_embs.transpose(0, 1)

        # 根据原论文的预测公式进行预测
        predictions = final_feat.matmul(item_embs)
        return predictions
