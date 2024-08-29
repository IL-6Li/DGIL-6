import numpy as np
import torch
from torch_geometric.data import DataLoader
import argparse
from models.GAT import GATModel
from utils.data_processing import load_data,load_bert_data
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, confusion_matrix,roc_curve
import torch.nn.functional as F
import matplotlib.pyplot as plt   
import pandas as pd
import joblib

def independent_test(args):

    threshold = args.d

    fasta_path_train_positive = args.pos_t
    fasta_path_val_positive = args.pos_v
    npz_dir_positive = args.pos_npz
    data_train, _ = load_bert_data(fasta_path_train_positive, npz_dir_positive, threshold, 1)
    data_val, _= load_bert_data(fasta_path_val_positive, npz_dir_positive, threshold, 1)

    data_train2, _ = load_data(fasta_path_train_positive, npz_dir_positive, threshold, 1)
    data_val2, _= load_data(fasta_path_val_positive, npz_dir_positive, threshold, 1)

    fasta_path_train_negative = args.neg_t
    fasta_path_val_negative = args.neg_v
    npz_dir_negative = args.neg_npz
    neg_data_train, _ = load_bert_data(fasta_path_train_negative, npz_dir_negative, threshold, 0)
    neg_data_val, _ = load_bert_data(fasta_path_val_negative, npz_dir_negative, threshold, 0)
    neg_data_train2, _ = load_data(fasta_path_train_negative, npz_dir_negative, threshold, 0)
    neg_data_val2, _ = load_data(fasta_path_val_negative, npz_dir_negative, threshold, 0)
    data_train.extend(neg_data_train)
    data_val.extend(neg_data_val)

    data_train2.extend(neg_data_train2)
    data_val2.extend(neg_data_val2)

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    model = torch.load(args.save).to(device)


    print(data_val2[0].x.shape)
    test_dataloader = DataLoader(data_val, batch_size=args.b,shuffle=False)
    model.eval()
    with torch.no_grad():
        total_num = 0
        total_correct = 0
        preds = []
        y_true = []
        y_pred = []
        arr_loss = []
        for data in test_dataloader:
            data = data.to(device)

            output =model(data.x,data.edge_index,data.batch)
            out = output[0].squeeze()
            score = torch.sigmoid(out)
            pred = (torch.sigmoid(out) >= th).to(torch.int)
            correct = (pred == data.y).sum().float()
            total_correct += correct
            total_num += data.num_graphs
            preds.extend(score.cpu().detach().data.numpy())
            y_true.extend(data.y.cpu().detach().data.numpy())
            y_pred.extend(torch.tensor(pred).cpu().detach().numpy().tolist())

        acc = (total_correct / total_num).item()
        auc = roc_auc_score(y_true, preds)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)
        bacc=(sn+sp)*0.5
     
        print("Test AUC: ", auc)
        print("BACC: ", bacc)
        print("ACC", acc)
        print("f1", f1)
        print("MCC", mcc)
        print("sn", sn)
        print("sp", sp)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument('-pos_t', type=str, default='/home/lqs/DGIL-6/data/train_data/positive/positive.fasta',
                        help='Path of the positive training dataset')
    parser.add_argument('-pos_v', type=str, default='/home/lqs/DGIL-6/data/test_data/positive/positive.fasta',
                        help='Path of the positive validation dataset')
    parser.add_argument('-pos_npz', type=str, default='/home/lqs/DGIL-6/data/train_data/positive/npz',
                        help='Path of the positive npz folder, which saves the predicted structure')

    parser.add_argument('-neg_t', type=str, default='/home/lqs/DGIL-6/data/train_data/negative/negative.fasta',
                        help='Path of the negative training dataset')
    parser.add_argument('-neg_v', type=str, default='/home/lqs/DGIL-6/data/test_data/negative/negitive.fasta', 
                        help='Path of the negative validation dataset')
    parser.add_argument('-neg_npz', type=str, default='/home/lqs/DGIL-6/data/train_data/negative/npz', 
                        help='Path of the positive npz folder, which saves the predicted structure')

    parser.add_argument('-b', type=int, default=256, help='Batch size')
    parser.add_argument('-save', type=str, default='/home/lqs/DGIL-6/saved_model',
                        help='The directory saving the trained models')

    parser.add_argument('-o', type=str, default='test_results.csv', help='Results file')
    parser.add_argument('-drop', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('-hd', type=int, default=64, help='Hidden layer dim')
    parser.add_argument('-heads', type=int, default=8, help='Number of heads')
    parser.add_argument('-d', type=int, default=0.8, help='Distance threshold')
    args = parser.parse_args()

    independent_test(args)
