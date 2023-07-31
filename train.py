import time
import pandas as pd
import torch
import argparse
from datetime import date
from utils import data_aug, evaluation, equipment, model_sel
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(model, dataset, index, bp, decay, bs, log, value, epoch, lr, relr_f, relr_p, min_loss):
    dates = date.today()
    device = equipment()
    model = model_sel(model, device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=value) if decay else Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=relr_f, patience=relr_p, verbose=False)
    train_l, val_l, test_l, train_d, val_d, test_d = data_aug(dataset, bs, device)
    con_str = f"{dates}-{model.__class__.__name__}-{dataset}-{decay}-{index}-"
    eva_sta = []  # Store criterion training-wise

    print('Model training started\nBatch size: %d\nLearning rate: %s\nNumber of epoch: %d' % (bs, lr, epoch), file=open(
          "./recording/" + str(con_str) + "log.txt", "w")) if log else print('Model training started\nBatch size: %d\n'
          'Learning rate: %s\nNumber of epoch: %d' % (bs, lr, epoch))
    since = time.time()

    for i in range(epoch):
        t_loss, v_loss, t_loss_b, v_loss_b, train_in, val_in = 0, 0, 0, 0, 0, 0
        t_tle, t_ple, v_tle, v_ple = [], [], [], []
        print('\nEpoch %d/%d \n' % (i + 1, epoch) + '-' * 60, file=open("./recording/" + str(con_str) + "log.txt",
              "a")) if log else print('\nEpoch %d/%d \n' % (i + 1, epoch) + '-' * 60)

        model.train()
        for step, (t_x, t_y) in enumerate(train_l):
            if str(device) == 'cuda': t_x, t_y = t_x.to(device), t_y.to(device)
            t_tle.append(t_y)
            output = model(t_x)
            loss = loss_func(output, t_y)
            lab = torch.argmax(output, 1)
            t_ple.append(lab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_loss_b += loss.item() * t_x.size(0)
            train_in += t_x.size(0)
            if bp: print("Train progress: %d/%d " % (train_in, len(train_d)), file=open("./recording/" + str(con_str) +
                         "log.txt", "a")) if log else print("Train progress: %d/%d " % (train_in, len(train_d)))

        t_loss = t_loss_b / len(train_d.targets)
        t_acc, t_npv, t_ppv, t_sen, t_spe, t_fos = evaluation(t_tle, t_ple)

        model.eval()
        for step, (v_x, v_y) in enumerate(val_l):
            if str(device) == 'cuda': v_x, v_y = v_x.to(device), v_y.to(device)
            v_tle.append(v_y)
            output = model(v_x)
            loss = loss_func(output, v_y)
            lab = torch.argmax(output, 1)
            v_ple.append(lab)
            v_loss_b += loss.item() * v_x.size(0)
            val_in += v_x.size(0)
            if bp: print("Validation progress: %d/%d " % (val_in, len(val_d)), file=open("./recording/" + str(con_str)
                         + "log.txt", "a")) if log else print("Validation progress: %d/%d " % (val_in, len(val_d)))

        v_loss = v_loss_b / len(val_d.targets)
        v_acc, v_npv, v_ppv, v_sen, v_spe, v_fos = evaluation(v_tle, v_ple)
        scheduler.step(v_loss)
        eva_sta_e = [t_loss, t_acc, t_npv, t_ppv, t_sen, t_spe, t_fos, v_loss, v_acc, v_npv, v_ppv, v_sen, v_spe, v_fos]
        eva_sta.append(eva_sta_e)

        t_c = time.time() - since
        print('Train and validation done in %d m %d s \nTrain loss: %.3f, acc: %.3f; Val loss: %.3f, acc: %.3f' % (
        t_c // 60, t_c % 60, t_loss, t_acc, v_loss, v_acc), file=open("./recording/" + str(con_str) + "log.txt", "a")) \
        if log else print('Train and validation done in %d m %d s \nTrain loss: %.3f, acc: %.3f; Val loss: %.3f, acc: '
                          '%.3f' % (t_c // 60, t_c % 60, t_loss, t_acc, v_loss, v_acc))

        if v_loss < min_loss:
            min_loss = v_loss
            torch.save(model, "./recording/" + str(con_str) + "model.pkl")
            if log: print("Model Saved", file=open("./recording/" + str(con_str) + "log.txt", "a"))
            else: print("Model Saved")

    df = pd.DataFrame(eva_sta)
    df.to_excel('./recording/' + str(con_str) + 'eva.xlsx', index=False, header=False)


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameters for train and validation process',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_argument_group()
    group.add_argument('--model', help='Choose your own model', choices=['CECT'], default='CECT')
    group.add_argument('--dataset', help='Select dataset', choices=['covid', 'radio', 'lungct'], default='covid')
    group.add_argument('--index', help='Index for number of run', required=True, metavar='INT')
    group.add_argument('--bp', help='Print progress batch-wise', default=False, metavar='BOOL')
    group.add_argument('--decay', help='Setting of weight decay', default=False, metavar='BOOL')
    group.add_argument('--bs', help='Batch size for training', default=256, type=int, metavar='INT')
    group.add_argument('--log', help='Save log to separate file', default=True, metavar='BOOL')
    group.add_argument('--value', help='Decay value', default=1e-3, type=float, metavar='FLOAT')
    group.add_argument('--epoch', help='Number of epochs', default=20, type=int, metavar='INT')
    group.add_argument('--lr', help='Learning rate', default=0.003, type=float, metavar='FLOAT')
    group.add_argument('--relr_f', help='Factor for learning rate decay', default=0.5, type=float, metavar='FLOAT')
    group.add_argument('--relr_p', help='Patience for learning rate decay', default=5, type=float, metavar='FLOAT')
    group.add_argument('--min_loss', help='Minimum loss for retrain', default=1e4, type=float, metavar='FLOAT')
    args = parser.parse_args()
    train(**vars(args))


if __name__ == '__main__':
    main()
