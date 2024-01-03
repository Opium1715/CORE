import argparse
import datetime
import os
import pickle as pkl
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CORE
from utils.dataloader import DataSet, compute_item_num, compute_max_len
from utils.callback import P_MRR_Record

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica',
                    help='dataset name: diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--random_seed', default=2023, help='random_seed')
parser.add_argument('--len_max', type=int, default=70, help='max lenghth of sequences')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--emb_size', type=int, default=100, help='hidden state size')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--dropout_tra', type=float, default=0.5)
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.5, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=4, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--valid_portion', type=float, default=0.1,
                    help='split the portion of training set as validation set')
parser.add_argument('--log_file', default='logs/', help='log dir path')
parser.add_argument('--shuffle', default=True)
parser.add_argument('--head_nums', type=int, default=2)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--mode', default='tra')
parser.add_argument('--tau', type=float, default=0.07)

opt = parser.parse_args()


def main():
    print(opt)
    set_seeds(opt.random_seed)
    print('设置种子：{}'.format(opt.random_seed))
    # log
    log_file = opt.log_file
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(log_file, 'log_' + time_str)
    metric_recorder = P_MRR_Record(save_path=log_dir, frequency=1)

    # load data
    train_data = pkl.load(open(f'dataset/{opt.dataset}/train.pkl', 'rb'))
    test_data = pkl.load(open(f'dataset/{opt.dataset}/test.pkl', 'rb'))
    all_data = pkl.load(open(f'dataset/{opt.dataset}/all_train_seq.pkl', 'rb'))
    item_num = compute_item_num(all_data)

    print('item_num:{}'.format(item_num))
    # create Dataset
    max_len = compute_max_len(train_data[0]) + 1
    print('max_len{}'.format(max_len))
    train_dataset = DataSet(rawData=train_data, max_len=max_len, train_mode=True)
    test_dataset = DataSet(rawData=test_data, max_len=max_len, train_mode=False)

    # create dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=2,
                                  drop_last=False)  # 直到被调用前，不会生成数据
    # for data in train_dataloader:
    #     print(data)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)

    # model
    core = CORE(item_num=item_num, emb_size=opt.emb_size, head_nums=opt.head_nums, dropout=opt.dropout,
                transformer_layers=opt.layers, encoder_mode=opt.mode, tau=opt.tau, dropout_attn=opt.dropout_tra,
                norm_eps=1e-12)
    # dgnn.compile()
    core.to(device=device)
    adam = torch.optim.Adam(core.parameters(), lr=opt.lr, weight_decay=opt.l2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=adam,
                                                   step_size=opt.lr_dc_step,
                                                   gamma=opt.lr_dc,
                                                   verbose=True)
    model_train(model=core,
                trainDataloader=train_dataloader,
                valDataloader=test_dataloader,
                loss_fn=torch.nn.CrossEntropyLoss(),
                optimizer=adam,
                scheduler=lr_scheduler,
                epoch=opt.epoch,
                recorder=metric_recorder)


def model_train(model, trainDataloader, valDataloader, loss_fn, optimizer, epoch, scheduler, recorder, **kwargs):
    # loss recorder
    epoch_loss = []
    best_pre = 0
    best_mrr = 0
    patience = 0
    for ep in range(1, epoch + 1):

        model.train()
        epoch_total_loss = 0.0
        print('Epoch ' + str(ep) + ':')
        with tqdm(total=len(trainDataloader), desc='train: ') as tbar:
            display_dict = {'loss': 0.0,
                            'avg_loss': 0.0}
            for session, label, mask in trainDataloader:
                session = session.to(device)
                label = label.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()
                score = model(session, mask)
                loss = loss_fn(score, label)
                # if loss > 100:
                #     print('损失异常记录{}'.format(loss))
                #     break
                # else:
                loss.backward()
                optimizer.step()
                display_dict['loss'] = loss.item()
                tbar.set_postfix(display_dict)
                # tbar.postfix['loss'] = loss.item()
                epoch_total_loss += loss.item()
                tbar.update(1)
            display_dict['avg_loss'] = epoch_total_loss / len(trainDataloader)
            tbar.set_postfix(display_dict)

        # validation
        model.eval()
        mrr = []
        precision = []
        with torch.no_grad():
            display_dict = {'P@20': 0.0, 'MRR@20': 0.0}
            with tqdm(total=len(valDataloader), postfix={}, desc='test: ') as tbar:
                for session, label, mask in valDataloader:
                    session = session.to(device)
                    label = label.to(device)
                    mask = mask.to(device)
                    score = model(session, mask)
                    # val
                    top_k_values, top_k_indices = torch.topk(score, k=20)
                    # precision
                    top_k_result = torch.stack([torch.isin(label[i], top_k_indices[i]) for i in range(label.shape[0])])
                    precision.append(top_k_result)
                    non_zero_num = torch.count_nonzero(top_k_result)
                    mrr.append(
                        torch.concat([1 / (torch.where(top_k_indices == label.unsqueeze(1))[1] + 1).to(torch.float64),
                                      torch.zeros((label.shape[0] - non_zero_num), device=device,
                                                  dtype=torch.float64)]))
                    tbar.update(1)
                precision = torch.mean(torch.concat(precision).to(torch.float64))
                mrr = torch.mean(torch.concat(mrr).to(torch.float64))
                display_dict['P@20'] = precision.item()
                display_dict['MRR@20'] = mrr.item()
                tbar.set_postfix(display_dict)
                recorder(precision.item(), mrr.item())
                # early_stop
                if precision.item() > best_pre:
                    best_pre = precision.item()
                    print('best precision:{}'.format(best_pre))
                if mrr.item() > best_mrr:
                    best_mrr = mrr.item()
                    print('best mrr:{}'.format(mrr))
                    patience = 0
                else:
                    patience += 1
                if patience >= opt.patience:
                    print('early stopping in epoch{}'.format(ep))
                    break
        scheduler.step()


def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = False  # 矩阵乘法的大小不是固定的，建议关闭乘法算法挑选
    torch.backends.cudnn.enabled = True


if __name__ == '__main__':
    main()
