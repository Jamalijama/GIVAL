import random
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_curve, auc, \
    precision_recall_curve, average_precision_score
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    #    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


#    torch.use_deterministic_algorithms(True)


def PrintTrain(x, train_loss_lst, train_acc_lst, file_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(x, train_loss_lst, 'r')
    ax[0].set_title('train loss')
    #    ax[0].set_xlabel('epochs')
    #    ax[0].set_ylabel('train loss')

    ax[1].plot(x, train_acc_lst, 'g')
    ax[1].set_title('valid accuracy')
    #    ax[1].set_xlabel('epochs')
    #    ax[1].set_ylabel('train accuracy')
    ax[1].set_ylim((0, 1.2))

    plt.savefig('./result/'+file_name, dpi=300)


def PrintROCPRAndCM(trues, preds, probs, pos_label, cm_file, roc_file, pr_file, roc_csv, pr_csv):
    #    sorted_indices = sorted(range(len(probs)), key=lambda k: probs[k], reverse=True)
    #    sorted_true = [true[i] for i in sorted_indices]
    #    sorted_probs = [probs[i] for i in sorted_indices]

    for i in range(len(trues)):
        plt.figure()
        true = trues[i]
        pred = preds[i]
        cm = confusion_matrix(true, pred)
        print(cm)
        cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #    sns.set_context("paper", font_scale=2)
        sns.set(font_scale=1)
        sns.heatmap(cm_prob, annot=True, fmt=".2f", cmap="Blues")
        #    plt.xlabel("Predicted")
        #    plt.ylabel("Actual")
        plt.savefig('./result/'+cm_file + '_' + str(i) + '.png', dpi=1000)
        plt.close()

        plt.figure()
        sns.set(font_scale=1)
#        df = pd.DataFrame()
        for i in range(len(trues)):
            true = trues[i]
            prob = probs[i]
            fpr, tpr, thresholds = roc_curve(true, prob, pos_label=pos_label)
            # auc = roc_auc_score(true, prob)
            roc_auc = auc(fpr, tpr)
#            df['fpr' + str(i)] = fpr
#            df['tpr' + str(i)] = tpr
            plt.plot(fpr, tpr, lw=2, label="ROC Curve (AUC = %0.4f)" % roc_auc)
           #    plt.xlabel("False Positive Rate")
           #    plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title('ROC Curve')
        plt.grid(False)
        plt.axes().set_facecolor('white')
        plt.savefig(roc_file, dpi=1000)
        plt.close()
#        df.to_csv(roc_csv, index=False)
    #
    # plt.figure()
    # sns.set(font_scale=2)
    # df = pd.DataFrame()
    # for i in range(len(trues)):
    #     true = trues[i]
    #     prob = probs[i]
    #     precision, recall, thresholds = precision_recall_curve(true, prob, pos_label=pos_label)
    #     auc_score = average_precision_score(true, prob)
    #     #    precision = precision[:-1]
    #     #    recall = recall[:-1]
    #     df['precision' + str(i)] = precision
    #     df['recall' + str(i)] = recall
    #
    #     plt.plot(recall, precision, lw=1.5, label='PR curve(AUC = %0.4f)' % auc_score)
    # plt.plot([0, 1], [1, 0], "k--")
    # #    plt.xlabel('Recall')
    # #    plt.ylabel('Precision')
    # plt.title('P-R Curve')
    # #    plt.xlim([0.0, 1.0])
    # #    plt.ylim([0.0, 1.0])
    # plt.legend(loc="lower right")
    # plt.savefig(pr_file, dpi=1000)
    # plt.close()
    # df.to_csv(pr_csv, index=False)


def Positive_probs(pos_label, pred, probs):
    positive_probs = []
    for i in range(len(pred)):
        if pred[i] == pos_label:
            positive_probs.append(probs[i][0])
        else:
            positive_probs.append(1 - probs[i][0])
    return positive_probs


# Calculate accuracy, loss, and gradient descent
def train(model, loader, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for input, label in loader:
        # input = torch.LongTensor(input)
        input = input.to(device)
        optimizer.zero_grad()
        predictions = model(input).to(device)
        #        print(predictions.cpu())
        #        print(label)
        loss = criterion(predictions.cpu(), label)
        acc = accuracy_score(predictions.argmax(1).cpu().detach().numpy(), label.cpu().detach().numpy())
        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)


# Calculate accuracy only
def evaluate(model, loader):
    model.eval()
    predictions = []
    true_labels = []
    probs = []
    with torch.no_grad():
        for input, label in loader:
            input = input.to(device)
            preds = model(input).to(device)
            prob = F.softmax(preds, dim=1)
            top_p, top_class = prob.topk(1, dim=1)
            probs.extend(top_p.cpu().detach().numpy().tolist())
            #            predictions.extend(top_class)
            predictions.extend(F.softmax(preds, dim=1).argmax(1).cpu().detach().numpy().tolist())
            true_labels.extend(label.cpu().detach().numpy().tolist())
    #            print('preds', preds)
    #            print('prob', prob)
    #            print('predictions', predictions[0])
    valid_acc = accuracy_score(true_labels, predictions)
    valid_f1 = f1_score(true_labels, predictions, average='macro')
    #    prec = precision_score(true_labels, predictions, average='macro')
    # rec = recall_score(true_labels, predictions, average='macro')
    return predictions, true_labels, probs, valid_acc, valid_f1


def test(model, loader):
    model.eval()
    predictions = []
    #    probabilities = []
    probs = []
    with torch.no_grad():
        for input in loader:
            input = input.to(device)
            preds = model(input).to(device)
            prob = F.softmax(preds, dim=1)
            # probs = []
            # for i in prob:
            #     max_value, _ = torch.max(i, dim=0)
            #     probs.append(max_value.item())
            top_p, top_class = prob.topk(1, dim=1)
            probs.extend(top_p)
            predictions.extend(top_class)
            # predictions.extend(F.softmax(preds, dim=1).argmax(1).cpu().detach().numpy().tolist())
            #            probabilities.extend(probs)

    return predictions, probs

def test_prob_all(model, loader):
    model.eval()
    predictions = []
    #    probabilities = []
    probs = []
    with torch.no_grad():
        for input in loader:
            input = input.to(device)
            preds = model(input).to(device)
            prob = F.softmax(preds, dim=1)
            # probs = []
            # for i in prob:
            #     max_value, _ = torch.max(i, dim=0)
            #     probs.append(max_value.item())

            top_p, top_class = prob.topk(1, dim=1)
            prob_all = prob
            #a = prob[0]
            #print(float(a[0]))
            probs.extend(prob_all)
            predictions.extend(top_class)
            #prob.cpu()
            prob = prob.tolist()
            # predictions.extend(F.softmax(preds, dim=1).argmax(1).cpu().detach().numpy().tolist())
            #            probabilities.extend(probs)
            
    return predictions, prob
