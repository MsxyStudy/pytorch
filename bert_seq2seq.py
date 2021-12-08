import argparse
import time
from typing import Optional
import torch
import pytorch_lightning as pl
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import logging
from torch.optim.lr_scheduler import StepLR
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#自定义数据集，Dataset是一个抽象类，自定义的数据集必须继承于这个抽象类
class MyDataset(Dataset):
    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

#数据预处理+装载数据
class RAMSData(pl.LightningDataModule):
    def __init__(self, argv):
        super(RAMSData, self).__init__()
        self.argv = argv
        self.batch_size = self.argv.batch_size
        self.mapping_dict = {}
        self.tokenizer = BertTokenizer.from_pretrained(self.argv.bert_path)

    #1、数据预处理
    #加载全局信息
    def prepare_data(self):
        #这个文件是什么，是官方的文件吗
        file = self.argv.event_role_multiplicities
        with open(file, mode='r', encoding='utf-8') as f:
            text = f.read().strip().split('\n')
            trigger_roles = {}
            triggers_type_to_ids = {}
            total_role_type = []
            for idx, item in enumerate(text):
                roles = []
                one_trigger_roles = item.split(' ')
                trigger = one_trigger_roles[0]
                triggers_type_to_ids[trigger] = idx + 1
                for i in range(1, len(one_trigger_roles), 2):
                    role = [one_trigger_roles[i], int(one_trigger_roles[i + 1])]
                    roles.append(role)
                    total_role_type.append('B-' + role[0])
                    total_role_type.append('E-' + role[0])
                trigger_roles[trigger] = roles
            total_role_type = list(sorted(set(total_role_type)))
            total_role_type.append('B-participant2')
            total_role_type.append('E-participant2')
            roles_type_to_ids = {'O': 0}
            for value, key in enumerate(total_role_type):
                roles_type_to_ids[key] = value + 1
            self.role_ids_to_type = [key for key in roles_type_to_ids.keys()]

        total_type_mask = {}
        for key in trigger_roles.keys():
            # 非角色类型为索引0
            mask = [1e10] * len(roles_type_to_ids)
            mask[0] = 0
            roles = trigger_roles[key]
            for item in roles:
                mask[roles_type_to_ids['B-' + item[0]]] = 0
                mask[roles_type_to_ids['E-' + item[0]]] = 0
            total_type_mask[key] = mask
        self.mapping_dict['trigger_roles'] = trigger_roles
        self.mapping_dict['triggers_type_to_ids'] = triggers_type_to_ids
        self.mapping_dict['roles_type_to_ids'] = roles_type_to_ids
        self.mapping_dict['total_type_mask'] = total_type_mask
    #从本地读取数据信息
    def load_data(self, file):
        raw_data = {}
        with open(file=file, mode='r', encoding='utf8') as f:
            sentences = []
            event = []
            doc_keys = []
            for line in f:
                data = json.loads(line)
                sentence = []
                doc_keys.append(data['doc_key'])
                for sen in data['sentences']:
                    # sentence += [word for word in sen]
                    sentence += [word.lower() for word in sen]
                sentences.append(sentence[:510])
                evt_triggers = data['evt_triggers']
                gold_evt_links = data['gold_evt_links']
                # triggers = [trigger, [idx]]
                triggers = [evt_triggers[0][2][0][0], evt_triggers[0][0:2]]
                # roles = [[role_1, [idx]], [role_2, [idx]], ...]
                roles = []
                for arg in gold_evt_links:
                    role = arg[2][11:]
                    idx = arg[1]
                    roles.append([role, idx])
                # event = [[triggers, roles], [triggers, roles], ...]
                event.append([triggers, roles])
            raw_data['sentences'] = sentences
            raw_data['event'] = event
            raw_data['doc_keys'] = doc_keys
        return raw_data
    #处理数据格式
    def processes_data(self, file, stage):
        sentences, event, doc_keys = self.load_data(file).values()
        trigger_roles, triggers_type_to_ids, roles_type_to_ids, total_type_mask = self.mapping_dict.values()
        data = []
        if stage in (None, "test"):
            self.test_info = []
        for j in tqdm(range(len(sentences))):
            sentence = sentences[j]
            e = event[j]
            doc_key = doc_keys[j]
            prepared_data = {}
            prepared_test_data = {}
            trigger, roles = e[0], e[1]

            # make src sentence and mapping between src sentence and tokenized sentence
            src_sentence = ['[CLS]'] + sentence + ['[SEP]']
            src_idx_to_tokenized_idx = torch.zeros(2, self.argv.max_src_length, dtype=torch.long)
            tokenized_src_sentence = []
            as_start = []
            as_end = []
            for i, word in enumerate(src_sentence):
                as_start.append(len(tokenized_src_sentence) if len(tokenized_src_sentence) < 512 else 511)
                tokenized_src_sentence += self.tokenizer.tokenize(word)
                as_end.append(len(tokenized_src_sentence) - 1 if len(tokenized_src_sentence) < 512 else 511)
            src_idx_to_tokenized_idx[0][:len(as_start)] = torch.tensor(as_start, dtype=torch.long)
            src_idx_to_tokenized_idx[1][:len(as_end)] = torch.tensor(as_end, dtype=torch.long)
            tokenized_idx_to_src_idx = {}
            if stage in (None, "test"):
                for r in range(src_idx_to_tokenized_idx.size()[1]):
                    start = src_idx_to_tokenized_idx[0][r]
                    end = src_idx_to_tokenized_idx[1][r]
                    for s in range(start, end + 1):
                        tokenized_idx_to_src_idx[s] = r

            # make trg sentence
            target_sen = []
            target_roles = trigger_roles[trigger[0]]
            for item in target_roles:
                target_sen += ['B-' + item[0], 'E-' + item[0]]
                if item[1] > 1:
                    target_sen += ['B-' + item[0] + '2', 'E-' + item[0] + '2']
            target_label_idx = {key: value for value, key in enumerate(target_sen)}

            # convert src sentence to ids
            encoded_src_sentence = self.tokenizer.encode_plus(tokenized_src_sentence, max_length=self.argv.max_src_length, truncation=True, padding='max_length', add_special_tokens=False)
            src_ids = torch.tensor(encoded_src_sentence['input_ids'])
            src_attention_mask = torch.tensor(encoded_src_sentence['attention_mask'])
            position_mask = torch.zeros(self.argv.max_src_length)
            position_mask[src_attention_mask.sum():] = 1e10

            # convert trg sentence to ids
            target_ids = torch.zeros(self.argv.max_target_length, dtype=torch.long)
            target_attention = torch.zeros(self.argv.max_target_length, dtype=torch.long)
            for i, item in enumerate(target_sen):
                target_ids[i] = roles_type_to_ids[target_sen[i]]
                target_attention[i] = 1

            # make output label
            target_label = torch.zeros(self.argv.max_target_length, dtype=torch.long)
            for item in roles:
                role_type = item[0]
                start_idx = item[1][0] + 1  # +1是因为有[CLS]标签
                end_idx = item[1][1] + 1
                new_start_idx = src_idx_to_tokenized_idx[0][start_idx]
                new_end_idx = src_idx_to_tokenized_idx[1][end_idx]
                target_label[target_label_idx['B-' + role_type]] = new_start_idx
                target_label[target_label_idx['E-' + role_type]] = new_end_idx
            # target_label[len(target_sen):] = -100

            # load data
            prepared_data['src_ids'] = src_ids
            prepared_data['src_attention_mask'] = src_attention_mask
            prepared_data['target_ids'] = target_ids
            prepared_data['target_attention'] = target_attention
            prepared_data['target_label'] = target_label
            prepared_data['position_mask'] = position_mask
            data.append(prepared_data)
            if stage in (None, "test"):
                prepared_test_data['doc_key'] = doc_key
                prepared_test_data['trigger'] = trigger
                prepared_test_data['tokenized_idx_to_src_idx'] = tokenized_idx_to_src_idx
                prepared_test_data['target_sen'] = target_sen
                self.test_info.append(prepared_test_data)
        return data
    #2、数据装载
    #将处理后的数据加载到MyDataset
    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_set = MyDataset(self.processes_data(self.argv.train_file, stage))
            self.val_set = MyDataset(self.processes_data(self.argv.dev_file, stage))
        if stage in (None, "test"):
            self.test_set = MyDataset(self.processes_data(self.argv.test_file, stage))
    #加载MyDataset数据到dataloader,方便模型读取数据
    def train_dataloader(self):
        train_loader = DataLoader(self.train_set, batch_size=self.argv.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_set, batch_size=self.argv.batch_size, shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_set, batch_size=self.argv.batch_size, shuffle=False)
        return test_loader

#自定义模型
class RAMSModule(pl.LightningModule):
    #定义相关的模型
    def __init__(self, argv):
        super(RAMSModule, self).__init__()
        self.argv = argv
        self.classification = 512
        # self.lstm = nn.LSTM(input_size=768, hidden_size=768, num_layers=1, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc1 = nn.Linear(768, self.classification)
        self.fc2 = nn.Linear(768, 768)
        self.fc3 = nn.Linear(768, 768)
        #相当于向量映射，将输入的句子向量化
        self.embedding = nn.Embedding(num_embeddings=133, embedding_dim=768)
        #在训练过程的前向传播中，让每个神经元以一定概率p处于不激活的状态，以达到减少过拟合的效果
        self.dropout = nn.Dropout(0.5)
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        self.bert = BertModel.from_pretrained(self.argv.bert_path)
        self.val_indicator = {'pred': [], 'true': [], 'loss': 0, 'count': 0}
        #pred:每个batch里面预测结果的集合
        #true：正确答案的集合
        #count：一个epoch里面batch的个数
        self.test_indicator = {'pred': [], 'true': [], 'loss': 0, 'count': 0}
    #前向传播，返回预测值
    def forward(self, src_ids, src_attention_mask, target_ids, position_mask):
        x = self.bert(input_ids=src_ids, attention_mask=src_attention_mask)[0]
        target_embedding = self.embedding(target_ids)
        target_embedding = F.relu(self.fc2(target_embedding))
        x = F.relu(self.fc3(x))
        y_hat = torch.einsum('ijk,ilk->ijl', target_embedding, x)
        # y_hat -= position_mask.unsqueeze(dim=1)
        return y_hat
    #优化器，用来更新参数
    def configure_optimizers(self):
        self.optimizer = AdamW(self.parameters(), lr=self.argv.learning_rate)
        self.lr_scheduler = StepLR(optimizer=self.optimizer, step_size=1, gamma=self.argv.gama)
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
    #训练的步骤，计算损失值
    def training_step(self, batch, batch_idx):
        src_ids, src_attention_mask, target_ids, target_attention, target_label, position_mask = batch.values()
        #self<==>self.forward
        y_hat = self(src_ids, src_attention_mask, target_ids, position_mask)
        loss = self.loss_function(y_hat.view((-1, self.classification)), target_label.view(-1))
        batch_size = src_attention_mask.size()[0]
        loss = loss.sum() / batch_size
        return loss
    #验证的步骤
    def validation_step(self, batch, batch_idx):
        src_ids, src_attention_mask, target_ids, target_attention, target_label, position_mask = batch.values()
        y_hat = self(src_ids, src_attention_mask, target_ids, position_mask)
        loss = self.loss_function(y_hat.view((-1, self.classification)), target_label.view(-1))
        batch_size = src_attention_mask.size()[0]
        loss = loss.sum() / batch_size
        return target_attention, y_hat, loss, target_label
    #batch结束后收集当前batch的数据
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.calculate_indicator(outputs, self.val_indicator)
    #计算前面所有收集数据的指标，用来评估模型
    def on_validation_epoch_end(self) -> None:
        self.display_indicator(self.val_indicator)
    #测试的步骤
    def test_step(self, batch, batch_idx):
        src_ids, src_attention_mask, target_ids, target_attention, target_label, position_mask = batch.values()
        y_hat = self(src_ids, src_attention_mask, target_ids, position_mask)
        loss = self.loss_function(y_hat.view((-1, self.classification)), target_label.view(-1))
        batch_size = src_attention_mask.size()[0]
        loss = loss.sum() / batch_size
        return target_attention, y_hat, loss, target_label

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.calculate_indicator(outputs, self.test_indicator)

    def on_test_epoch_end(self) -> None:
        test_info = self.trainer.datamodule.test_info
        total_pred = self.test_indicator['pred']
        with open('./RAMS_1.0/scorer/pred_outputs.jsonlines', mode='w', encoding='utf-8') as f:
            for i in range(len(test_info)):
                doc_key, trigger, tokenized_idx_to_src_idx, target_sen = test_info[i].values()
                target_type = []
                for k in range(0, len(target_sen), 2):
                    if target_sen[k] == 'B-participant2':
                        target_type.append('participant')
                    else:
                        target_type.append(target_sen[k][2:])
                pred = total_pred[i]
                extract = {'doc_key': doc_key, 'predictions': []}
                event = [trigger[1]]
                for j in range(len(pred)):
                    if 0 < pred[j][0] <= pred[j][1]:
                        real_start = tokenized_idx_to_src_idx[pred[j][0]]
                        real_end = tokenized_idx_to_src_idx[pred[j][1]]
                        event.append([real_start - 1, real_end - 1, target_type[j], 1.0])
                extract['predictions'].append(event)
                json_info = json.dumps(extract)
                f.write(json_info + '\n')
        self.display_indicator(self.test_indicator)
    #收集每个batch数据
    def calculate_indicator(self, outputs, indicator):
        target_attention, y_hat, loss, target_label = outputs
        batch_size = target_attention.size()[0]
        pred = y_hat.argmax(dim=-1)
        for i in range(batch_size):
            valid_nums = target_attention[i].sum()
            tmp_true = []
            tmp_pred = []
            for j in range(0, valid_nums, 2):
                tmp_pred.append([pred[i][j].item(), pred[i][j + 1].item()])
                tmp_true.append([target_label[i][j].item(), target_label[i][j + 1].item()])
            indicator['pred'].append(tmp_pred)
            indicator['true'].append(tmp_true)
        indicator['loss'] += loss
        indicator['count'] += 1
    #计算指标
    def display_indicator(self, indicator):
        right_nums = 0
        pred_nums = 0
        true_nums = 0
        true = indicator['true']
        pred = indicator['pred']
        for i in range(len(true)):
            for j in range(len(true[i])):
                if pred[i][j][0] != 0 and pred[i][j][1] != 0:
                    pred_nums += 1
                if true[i][j][0] != 0 and true[i][j][1] != 0:
                    true_nums += 1
                    if pred[i][j][0] == true[i][j][0] and pred[i][j][1] == true[i][j][1]:
                        right_nums += 1
        precision = right_nums / (pred_nums + 1e-8)
        recall = right_nums / (true_nums + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        loss = indicator['loss']
        count = indicator['count']
        loss /= count
        logger.info('precision: {:.3f}, recall: {:.3f}, f1: {:.3f}, loss: {:.3f}'.format(precision, recall, f1, loss))
        self.log('f1', f1)
        indicator['true'] = []
        indicator['pred'] = []
        indicator['loss'] = 0
        indicator['count'] = 0


def main(is_train):
    #ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息
    parser = argparse.ArgumentParser()
    #调用add_argument()给ArgumentParser对象添加程序所需的参数信息
    #输入句子的最大长度
    parser.add_argument("--max_src_length", default=512, help="max length of the input sentences")
    parser.add_argument("--bert_path", default='./bert-base-uncased')
    parser.add_argument("--event_role_multiplicities", default='./RAMS_1.0/scorer/event_role_multiplicities.txt')
    parser.add_argument("--checkpoint_path", default='./checkpoint')
    #训练集、验证集、测试集的文件路径
    parser.add_argument("--train_file", default='./RAMS_1.0/data/train.jsonlines', help="file path of train data")
    parser.add_argument("--dev_file", default='./RAMS_1.0/data/dev.jsonlines', help="file path of dev data")
    parser.add_argument("--test_file", default='./RAMS_1.0/data/test.jsonlines', help="file path of test data")
    parser.add_argument("--batch_size", default=4, help="size of each batch")
    #
    parser.add_argument("--max_target_length", default=10)
    #训练次数
    parser.add_argument("--max_epochs", default=5, help="nums of epoch")
    parser.add_argument("--nums_gpus", default=0, help="use gpus")
    parser.add_argument("--gama", default=0.1)
    parser.add_argument("--learning_rate", default=1e-4)
    #将数据封装到argv中，通过argv对象来调用其成员变量
    argv = parser.parse_args()
    #将对象传入数据集
    dm = RAMSData(argv=argv)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filename='{epoch:02d}-{f1:.3f}', monitor="f1", mode="max",
                                                       save_top_k=3)
    if argv.nums_gpus > 1:
        trainer = pl.Trainer(max_epochs=argv.max_epochs, gpus=argv.nums_gpus, callbacks=[checkpoint_callback],
                             strategy="ddp_spawn")
    else:
        trainer = pl.Trainer(max_epochs=argv.max_epochs, gpus=argv.nums_gpus, callbacks=[checkpoint_callback],
                             gradient_clip_val=1.0)
    # =====train and validation=====
    if is_train:
        model = RAMSModule(argv=argv)
        trainer.fit(model=model, datamodule=dm)
        # trainer.test(datamodule=dm, ckpt_path='best')
    else:
        # =====only test=====
        model = RAMSModule.load_from_checkpoint(argv=argv,
                                                checkpoint_path='./lightning_logs/version_1/checkpoints/epoch=04-f1=0.377.ckpt')
        trainer.test(model=model, datamodule=dm)

#函数入口
if __name__ == '__main__':
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # 训练数据
    is_train = True
    # is_train = False
    main(is_train)

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end))
