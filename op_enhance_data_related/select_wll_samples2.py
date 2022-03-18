# author:dgl
# datetime:2022/2/10 3:58 PM

"""
文件说明：
"""
import copy
import itertools
import os

import numpy as np
import torch
import tqdm
from sklearn.linear_model import LogisticRegressionCV
from torch.utils.data import DataLoader
from torch.utils.data.dataset import T_co

import sys
sys.path.append("X:/Python/op_enhance")
from estimate_wrong_samples.sa import group_teacher_ats, compute_dsa, compute_lsa
from estimate_wrong_samples.uncertainty_lcr_kde import compute_uncertainty_lcr, compute_kde, mp
from estimate_wrong_samples.model_mutation.operator import OpType
from utils.commons import extract_prediction_info
from utils.time_utils import current_timestamp, folder_timestamp
from adv_mutants.pgd import projected_gradient_descent
from op_craft.op_data_factory import MySimpleDataset
from sklearn.preprocessing import scale
from estimate_wrong_samples.performance_estimate.ces_esitamate import estimate_op_acc_with_ces


def get_precision(indices, preds, true_labels):
    return sum(preds[i] != true_labels[i] for i in indices) / len(indices)


def select_wl_indices(dsa, lcr, cfdc, total_wl, preds=None, true_labels=None):
    dsa_sorted_idx = np.argsort(dsa)[::-1][:total_wl]
    lcr_sorted_idx = np.argsort(lcr)[::-1][:total_wl]
    cfdc_sorted_idx = np.argsort(cfdc)[:total_wl]

    all_idx_cnt = np.bincount(np.concatenate((dsa_sorted_idx, lcr_sorted_idx, cfdc_sorted_idx)))
    selected_indices = np.where(all_idx_cnt == 3)[0]
    if preds is not None and true_labels is not None:
        precison = get_precision(selected_indices, preds, true_labels)
        print(f"WL select precision:{precison:.4f}")
    return selected_indices


def compute_presision(dsa, uc, lcr, cfdc, total_wl, preds, true_labels):
    dsa_sorted_idx = np.argsort(dsa)[::-1][:total_wl]
    uc_sorted_idx = np.argsort(uc)[::-1][:total_wl]
    lcr_sorted_idx = np.argsort(lcr)[::-1][:total_wl]
    cfdc_sorted_idx = np.argsort(cfdc)[:total_wl]

    dsa_pre = get_precision(dsa_sorted_idx, preds, true_labels)
    uc_pre = get_precision(uc_sorted_idx, preds, true_labels)
    lcr_pre = get_precision(lcr_sorted_idx, preds, true_labels)
    cfdc_pre = get_precision(cfdc_sorted_idx, preds, true_labels)

    temp_1 = np.bincount(np.concatenate((dsa_sorted_idx, uc_sorted_idx, cfdc_sorted_idx)))
    vote_res_1 = np.where(temp_1 == 3)[0]
    comb_pre_1 = get_precision(vote_res_1, preds, true_labels)

    temp_2 = np.bincount(np.concatenate((dsa_sorted_idx, lcr_sorted_idx, cfdc_sorted_idx)))
    vote_res_2 = np.where(temp_2 == 3)[0]
    comb_pre_2 = get_precision(vote_res_2, preds, true_labels)

    dsa_pre1 = get_precision(dsa_sorted_idx[:len(vote_res_1)], preds, true_labels)
    uc_pre1 = get_precision(uc_sorted_idx[:len(vote_res_1)], preds, true_labels)
    lcr_pre1 = get_precision(lcr_sorted_idx[:len(vote_res_1)], preds, true_labels)
    cfdc_pre1 = get_precision(cfdc_sorted_idx[:len(vote_res_1)], preds, true_labels)

    dsa_pre2 = get_precision(dsa_sorted_idx[:len(vote_res_2)], preds, true_labels)
    uc_pre2 = get_precision(uc_sorted_idx[:len(vote_res_2)], preds, true_labels)
    lcr_pre2 = get_precision(lcr_sorted_idx[:len(vote_res_2)], preds, true_labels)
    cfdc_pre2 = get_precision(cfdc_sorted_idx[:len(vote_res_2)], preds, true_labels)

    print(f"comb_pre_uc:{comb_pre_1:.4f}({len(vote_res_1)}), comb_pre_lcr:{comb_pre_2:.4f}({len(vote_res_2)})")
    print(f"dsa_pre:{dsa_pre:.4f}({total_wl}), {dsa_pre1}({len(vote_res_1)}), {dsa_pre2}({len(vote_res_2)})")
    print(f"lcr_pre:{lcr_pre:.4f}({total_wl}), {lcr_pre1}({len(vote_res_1)}), {lcr_pre2}({len(vote_res_2)})")
    print(f"uc_pre :{uc_pre:.4f}({total_wl}), {uc_pre1}({len(vote_res_1)}), {uc_pre2}({len(vote_res_2)})")
    print(f"cfdc_pre:{cfdc_pre:.4f}({total_wl}),{cfdc_pre1}({len(vote_res_1)}), {cfdc_pre2}({len(vote_res_2)})")


def extract_features(args, seed_model, train_at_buckes, test_data_laoder, test_ats, test_preds, device):
    # print(f"{current_timestamp()}: comput kde...")
    # kde = compute_kde(train_at_buckes, test_ats, test_preds, args.bandwidth)
    # print(f"{current_timestamp()}: comput lsa...")
    # lsa = compute_lsa(train_at_buckes, test_ats, test_preds)
    print(f"{current_timestamp()}: comput dsa...")
    dsa = compute_dsa(train_at_buckes, test_ats, test_preds, device)
    print(f"{current_timestamp()}: comput uncertainty ans lcr...")
    uc, lcr = compute_uncertainty_lcr(seed_model, args.operator, args.mutation_ration,
                                      test_data_laoder,
                                      test_preds, args.nb_mutants,
                                      "cuda:2")

    return dsa, uc, lcr

#seed_model: resnet18, args:GF NAI NS WS  mutation_ration=1/100,3/1000,3/10000.... nb_mutants=100,200,300MAX
#test_preds=predict labels

def select_wll_samples_lcr(args, seed_model, test_data_laoder, test_preds, threshold):
    uc, lcr = compute_uncertainty_lcr(seed_model, args.operator, args.mutation_ration,
                                      test_data_laoder,
                                      test_preds, args.nb_mutants,
                                      "cuda:2")
    torch.save(lcr, "lcr.pt")
    for i in lcr:
        print(i)
    wl_indices = np.where(np.asarray(lcr) > threshold)[0]
    return wl_indices



def select_wll_samples(args, seed_model, ori_train_dataset, op_dataset, layer_names, is_rq2=True):
    """

    Args:
        args ():
        seed_model ():
        ori_train_dataset ():
        op_dataset ():
        layer_names ():

    Returns:
        wl_indices, wl_samples, preds, true_lables

    """
    device = args.device
    train_data_loader = DataLoader(dataset=ori_train_dataset, batch_size=32)
    op_data_laoder = DataLoader(dataset=op_dataset, batch_size=32)

    print(f"{current_timestamp()}: teacher ats...")
    train_preds, train_cfdc, train_true_labels, train_ats = extract_prediction_info(copy.deepcopy(seed_model),
                                                                                    train_data_loader,
                                                                                    layer_names, device)

    print(f"{current_timestamp()}: op ats...")
    op_preds, op_cfdc_list, op_true_labels, op_ats = extract_prediction_info(copy.deepcopy(seed_model),
                                                                             op_data_laoder,
                                                                             layer_names,
                                                                             device)

    train_at_buckes = group_teacher_ats(train_ats, train_preds, train_preds, [1] * len(train_preds), 0)

    print("extracting op features...")
    dsa, uc, lcr = extract_features(args, seed_model, train_at_buckes, op_data_laoder, op_ats, op_preds, device)

    est_acc, current_indices = estimate_op_acc_with_ces(op_ats,
                                                        op_preds,
                                                        op_true_labels,
                                                        budgets=args.est_budgets,
                                                        max_iter=args.est_nit,
                                                        batch_size=args.est_batch,
                                                        p=args.est_init)
    total_wl = int(len(op_preds) * (1 - est_acc))
    print(f"{current_timestamp()} estimate acc: {est_acc:.4f}")
    # save_path = f"cached_data/{(folder_timestamp())}"
    # torch.save([dsa, uc, lcr, op_cfdc_list, total_wl, op_preds, op_true_labels], save_path)
    # print(f"saved path:{save_path}")
    if is_rq2:
        compute_presision(dsa, uc, lcr, op_cfdc_list, total_wl, op_preds, op_true_labels)
    else:
        if args.verbose:
            wl_indices = select_wl_indices(dsa, lcr, op_cfdc_list, total_wl, op_preds, op_true_labels)
        else:
            wl_indices = select_wl_indices(dsa, lcr, op_cfdc_list, total_wl)

        # wl_samples = []
        # wl_preds = []
        # wl_true_labels = []
        # for i in wl_indices:
        #     wl_samples.append(op_dataset[i][0])
        #     wl_true_labels.append(op_dataset[i][1])
        #     wl_preds.append(op_preds[i])
        # return wl_indices, wl_samples, wl_preds, wl_true_labels
        return wl_indices, op_preds


def _voting_analysis(dsa, uc, lcr, cfdc_list, number_wrong, preds, true_labels):
    def get_precision(indices):
        if len(indices) == 0:
            return None
        return sum(preds[i] != true_labels[i] for i in indices) / len(indices)

    dsa_sorted_idx = np.argsort(dsa)[::-1][:number_wrong]
    lcr_sorted_idx = np.argsort(lcr)[::-1][:number_wrong]
    uc_sorted_idx = np.argsort(uc)[::-1][:number_wrong]
    # kde_sorted_idx = np.argsort(kde)[:number_wrong]
    cfdc_sorted_idx = np.argsort(cfdc_list)[:number_wrong]

    # only one
    print("dsa", get_precision(dsa_sorted_idx))
    print("lcr", get_precision(lcr_sorted_idx))
    print("uc", get_precision(uc_sorted_idx))
    # print("kde", get_precision(kde_sorted_idx))
    print("cfdc", get_precision(cfdc_sorted_idx))
    print("======================================")

    if cfdc_list is not None:
        all_idx = np.concatenate((dsa_sorted_idx, lcr_sorted_idx, uc_sorted_idx, cfdc_sorted_idx))
    all_idx = np.concatenate((dsa_sorted_idx, lcr_sorted_idx, uc_sorted_idx))

    res = np.bincount(all_idx)
    # if cfdc_list is not None:
    #     vote5res = np.where(res == 5)[0]
    vote4res = np.where(res == 4)[0]
    vote3res = np.where(res == 3)[0]
    vote2res = np.where(res == 2)[0]
    # if cfdc_list is not None:
    #     vote5pre = get_precision(vote5res)
    vote4pre = get_precision(vote4res)
    vote3pre = get_precision(vote3res)
    vote2pre = get_precision(vote2res)
    # if cfdc_list is not None:
    #     print("vote5", vote5pre, len(vote5res))
    print("vote4", vote4pre, len(vote4res))
    print("vote3", vote3pre, len(vote3res))
    print("vote2", vote2pre, len(vote2res))
    print("======================================")

    # only two
    from itertools import combinations
    for cb in combinations(["dsa", "lcr", "uc", "cfdc"], 2):
        idx1 = eval(f"{cb[0]}_sorted_idx")
        idx2 = eval(f"{cb[1]}_sorted_idx")
        temp = np.bincount(np.concatenate((idx1, idx2)))
        vote_res = np.where(temp == 2)[0]
        print(cb, get_precision(vote_res), len(vote_res))

    print("=================Three=====================")
    # only three
    for cb in combinations(["dsa", "lcr", "uc", "cfdc"], 3):
        idx1 = eval(f"{cb[0]}_sorted_idx")
        idx2 = eval(f"{cb[1]}_sorted_idx")
        idx3 = eval(f"{cb[2]}_sorted_idx")
        temp = np.bincount(np.concatenate((idx1, idx2, idx3)))
        vote_res = np.where(temp == 3)[0]
        print(cb, get_precision(vote_res), len(vote_res))
    print("=================Four=====================")
    for cb in combinations(["dsa", "lcr", "uc", "cfdc"], 4):
        idx1 = eval(f"{cb[0]}_sorted_idx")
        idx2 = eval(f"{cb[1]}_sorted_idx")
        idx3 = eval(f"{cb[2]}_sorted_idx")
        idx4 = eval(f"{cb[3]}_sorted_idx")
        temp = np.bincount(np.concatenate((idx1, idx2, idx3, idx4)))
        vote_res = np.where(temp >= 2)[0]
        print(cb, get_precision(vote_res), len(vote_res))


def lr(normal_features_tuple, adv_features_tuple, op_features_tuple, op_preds, op_true_labels):
    normal_values = []
    adv_values = []
    op_values = []
    metrics = ["dsa", "uc", "lcr", "kde"]
    for cb in itertools.combinations(np.arange(4), 2):
        print(">>>>>>>>>>>>>>>>>>>>>")
        print([metrics[i] for i in cb])
        for i in cb:
            normal_values.append(normal_features_tuple[i])
            adv_values.append(adv_features_tuple[i])
            op_values.append(op_features_tuple[i])
        # normal_values = (normal_features_tuple[0], normal_features_tuple[2])
        # adv_values = (adv_features_tuple[0], adv_features_tuple[2])
        # op_values = (op_features_tuple[0], op_features_tuple[2])
        # normal_values = normal_features_tuple
        # adv_values = adv_features_tuple
        # op_values = op_features_tuple

        normal_features = np.array(normal_values).transpose()
        adv_features = np.array(adv_values).transpose()
        values = np.concatenate((normal_features, adv_features))
        neg_labels = np.zeros(normal_features.shape[0], dtype=int)
        pos_labels = np.ones(adv_features.shape[0], dtype=int)
        labels = np.concatenate((neg_labels, pos_labels))

        op_features = np.array(op_values).transpose()
        all_data = scale(np.concatenate((values, op_features)))
        values = all_data[:values.shape[0]]
        op_features = all_data[values.shape[0]:]
        lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)
        lr_predicts = lr.predict(values)
        train_acc = len(np.where(lr_predicts == labels)[0]) / len(labels)

        ############
        #
        ############
        op_lr_preds = lr.predict(op_features)
        op_true_wl_labels = op_preds != op_true_labels
        test_acc = len(np.where(op_true_wl_labels == op_lr_preds)[0]) / len(op_preds)

        pred_wl_indices = np.where(op_lr_preds == 1)[0]
        true_wl_indices = np.where(op_preds != op_true_labels)[0]
        precision = len(set(pred_wl_indices) & set(true_wl_indices)) / len(pred_wl_indices)
        print("estimated acc", 1 - len(pred_wl_indices) / len(op_preds))
        print(f"classifier train acc {train_acc:.4f}, test acc {test_acc:.4f}, precision {precision:.4f}")


def voting_analysis(timeid):
    # timeid = "20220216153445"
    # timeid = "20220216194137"
    # timeid = "20220216170803"

    # ori_train_dataset, op_dataset, normal_samples, adv_samples = torch.load("../RQ/processed_data.pt")
    normal_features_tuple, adv_features_tuple, op_features_tuple = torch.load(
        f"../RQ/cached_data/{timeid}/features.pt")
    op_cfdc_list, op_preds, op_true_labels = torch.load(f"../RQ/cached_data/{timeid}/op.pt")

    op_acc = len(np.where(op_preds == op_true_labels)[0]) / len(op_preds)
    print(f"op acc:{op_acc}")
    number_wrong = int(len(op_preds) * (1 - op_acc))
    print(f"number_wrong:{number_wrong}")
    # lr(normal_features_tuple, adv_features_tuple, op_features_tuple, op_preds, op_true_labels)
    print(f"================={timeid}=============================")
    _voting_analysis(op_features_tuple, number_wrong, op_preds, op_true_labels, cfdc_list=op_cfdc_list)


if __name__ == '__main__':
    # analysis()
    # for timeid in ["20220216150813", "20220216151049", "20220216153445", "20220216170803", "20220216194137"]:
    #     voting_analysis(timeid)
    dsa, uc, lcr, op_cfdc_list, total_wl, op_preds, op_true_labels = torch.load(f"../RQ/cached_data/20220222113108")
    print(lcr)
    print(len(lcr))
    print(len(np.where(np.asarray(lcr) > 0.02)[0]))
    # _voting_analysis(dsa, uc, lcr, op_cfdc_list, total_wl, op_preds, op_true_labels)
