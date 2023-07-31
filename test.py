import time
import torch
import argparse
from utils import data_aug, confusion, evaluation, equipment


def test(model, dataset, index, bp, decay, bs, td):
    device = equipment()
    loss_func = torch.nn.CrossEntropyLoss()
    train_l, val_l, test_l, train_d, val_d, test_d = data_aug(dataset, bs, device)
    con_str = f"{td}-{model}-{dataset}-{decay}-{index}-"
    model = torch.load("./recording/" + str(con_str) + "model.pkl") if str(device) == 'cuda' else torch.load(
                       "./recording/" + str(con_str) + "model.pkl", map_location=torch.device('cpu'))
    te_loss, te_loss_b, te_in = 0, 0, 0
    te_tle, te_ple = [], []

    print('Model testing started', file=open("./recording/" + str(con_str) + "test.txt", "a"))
    since = time.time()
    model.eval()
    for step, (t_x, t_y) in enumerate(test_l):
        if str(device) == 'cuda': t_x, t_y = t_x.to(device), t_y.to(device)
        te_tle.append(t_y)
        output = model(t_x)
        loss = loss_func(output, t_y)
        lab = torch.argmax(output, 1)
        te_ple.append(lab)
        te_loss_b += loss.item() * t_x.size(0)
        te_in += t_x.size(0)
        if bp: print("Test progress: %d/%d " % (te_in, len(test_d)))

    t_c = time.time() - since
    te_loss = te_loss_b / len(test_d.targets)
    te_acc, te_npv, te_ppv, te_sen, te_spe, te_fos = evaluation(te_tle, te_ple)
    confusion(con_str, te_tle, te_ple)
    print('Test done in %d m %d s \nTest loss: %.3f, acc: %.3f, npv: %.3f, ppv: %.3f, sen: %.3f, spe: %.3f, fos: %.3f'
          % (t_c // 60, t_c % 60, te_loss, te_acc, te_npv, te_ppv, te_sen, te_spe, te_fos), file=open("./recording/" +
          str(con_str) + "test.txt", "a"))


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameters for test process',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    group = parser.add_argument_group()
    group.add_argument('--model', help='Choose your own model', choices=['CECT'], default='CECT')
    group.add_argument('--dataset', help='Select dataset', choices=['covid', 'radio'], default='covid')
    group.add_argument('--index', help='Index for number of run', required=True, metavar='INT')
    group.add_argument('--bp', help='Print progress batch-wise', default=False, metavar='BOOL')
    group.add_argument('--decay', help='Setting of weight decay', default=False, metavar='BOOL')
    group.add_argument('--bs', help='Batch size for testing', default=32, type=int, metavar='INT')
    group.add_argument('--td', help='Training date with format YYYY-MM-DD', required=True, metavar='STR')
    args = parser.parse_args()
    test(**vars(args))


if __name__ == '__main__':
    main()
