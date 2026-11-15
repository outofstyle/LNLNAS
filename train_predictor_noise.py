# -*- coding: utf-8 -*-
# pylint: disable-all

"""
Train architecture-performance predictor on NAS-Bench-201 / NAS-Bench-301 / NDS-ResNet/ResNeXT-A.

First, pretrain the predictor on a single type of low-fidelity information.
Second, finetune the predictor on the actual performance data.
"""

import os
import sys
import logging
import shutil
import argparse
import pickle
import yaml

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import setproctitle
from torch.utils.data import DataLoader

from dele import utils
from dele.dataset import LFArchDataset, Index_LFArchDataset
from dele.arch_embedder import NasBench201_LSTMSeqEmbedder
from dele.arch_network import PointwiseComparator, NoiseComparator


def warm_up(train_loader, model, epoch, args, use_low_fidelity: bool = False):
    
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()

    model.train()
    for step, (archs, real_accs, low_fidelity, index) in enumerate(train_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)

        if args.compare:
            
            archs_1, archs_2, better_lst = compare_data(
                archs, 
                index, 
                accs, 
                args
            )
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            loss = model.update_compare_warm_up(archs_1, archs_2, better_lst, pen_weight = args.warmup_pen_weight, js_weight = args.warmup_js_weight)
            
        
                
            objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

        
        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if args.compare else ""))

    return objs.avg



def eval_sample(val_loader, model, use_low_fidelity=True, k=32, alpha=0.2,
                sigma=1.0, selection_ratio=0.015, semi_topk=2048,
                pair_weight=1.0, mix_weight=1.0, max_iter=10, reg_weight=5e-4,
                seed=None, timings=None):
    """Filter every predictor's loss matrix, returning two K-element lists.

    filtered_pair_sets[r] contains the pairs selected by predictor r.
    semi_pair_sets[r] contains that predictor's top semi_topk pairs.
    Dense loss matrices are consumed one at a time, rather than returned.
    """
    if len(val_loader) != 1:
        raise ValueError("eval_sample requires one full-dataset batch")
    if sigma <= 0 or not 0 < selection_ratio <= 1:
        raise ValueError("Require sigma > 0 and 0 < selection_ratio <= 1")
    if semi_topk < 0:
        raise ValueError("semi_topk must be nonnegative")
    model.eval()
    filtered_pair_sets, semi_pair_sets = [], []
    with torch.no_grad():
        archs, real_accs, proxy_scores, indices = next(iter(val_loader))
        archs = np.asarray(archs)
        scores = np.asarray(proxy_scores if use_low_fidelity else real_accs,
                            dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        kwargs = dict(alpha=alpha, k=k, pair_weight=pair_weight, mix_weight=mix_weight)
        if hasattr(model, "iter_sample_losses"):
            matrices = model.iter_sample_losses(archs, scores, **kwargs)
        else:
            # MultiComparator exposes a tuple of per-predictor loss matrices.
            def legacy_matrices():
                yield from model.compute_sample_losses(archs, scores, **kwargs)
            matrices = legacy_matrices()
        branch = 0
        try:
            while True:
                synchronize()
                start = time.perf_counter()
                try:
                    matrix = next(matrices)
                except StopIteration:
                    break
                synchronize()
                if timings is not None:
                    timings["pair_construction_seconds"] += time.perf_counter() - start
                start = time.perf_counter()
                filtered, semi = select_pairs(
                    matrix, indices, sigma, selection_ratio, seed,
                    max_iter, reg_weight, semi_topk=semi_topk)
                if timings is not None:
                    timings["pair_filtering_seconds"] += time.perf_counter() - start
                filtered_pair_sets.append(filtered)
                semi_pair_sets.append(semi)
                logging.info("Predictor %d selected %d pairs", branch, len(filtered))
                del matrix
                branch += 1
        finally:
            if hasattr(matrices, "close"):
                matrices.close()
    expected = getattr(model, "num_predictors", len(filtered_pair_sets))
    if not filtered_pair_sets or len(filtered_pair_sets) != expected:
        raise ValueError("The number of loss matrices does not match the predictor count")
    return filtered_pair_sets, semi_pair_sets


def pre_train_1(
    train_loader, 
    high_quality_pair1,
    high_quality_pair2,
    model,
    epoch: int,
    args, 
    use_low_fidelity: bool = False,
   
):
    """
    Train the vanilla predictor.

    Args:
        train_loader: Training data loader.
        model (PointwiseComparator): The vanilla predictor.
        epoch (int): Current epoch number.
        compare (bool): If `True`, train the predictor with ranking loss.
                        Else if `False`, train the predictor with regression loss.
                        Default: `True`.
        use_low_fidelity (bool): Whether train the predictor on the low-fidelity information.
                                 Default: `False`.
        max_compare_ratio (float): Default: 4.
        compare_threshold (float): Default: 0.
        choose_pair_criterion (str): Default: "random".

    Returns:
        The training loss (float).

    """
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()

    model.train()
    for step, (archs, real_accs, low_fidelity, index) in enumerate(train_loader):
        #archs = np.array(archs)
       
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs

        n = len(archs)
    

        if args.compare:
            all_archs_11, all_archs_12, all_better_lst1, all_index_11, all_index_12 = batch_process_pairs(
                archs, 
                index, 
                accs, 
                high_quality_pair1,
                512
            )
          

            all_archs_21, all_archs_22, all_better_lst2, all_index_21, all_index_22 = batch_process_pairs(
                archs, 
                index, 
                accs, 
               high_quality_pair2,
               512
            )
            
            n_diff_pairs = len(all_better_lst1[0])
            n_diff_pairs_meter.update(float(n_diff_pairs))
            num_batch = len(all_archs_11)
            for k in range(num_batch):
                archs_11 = np.array(all_archs_11[k])
                archs_12 = np.array(all_archs_12[k])
                archs_21 = np.array(all_archs_21[k])
                archs_22 = np.array(all_archs_22[k])
                better_lst1 = all_better_lst1[k]
                better_lst2 = all_better_lst2[k]
                loss = model.cal_semi_loss_retrain( archs_11, archs_12, better_lst1,  archs_21, archs_22, better_lst2, sim_weight = args.pretrain_sim_weight, bt_weight = args.pretrain_bt_weight, lambda_weight = args.pretrain_lambda_weight)
            
                objs.update(loss, n_diff_pairs)
        else:
            loss = model.update_predict(archs, low_fidelity)
            objs.update(loss, n)

        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(train_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if args.compare else ""))
    
    return objs.avg



def train(real_loader, all_data_loader, high_quality_pair1,
        high_quality_pair2, model, epoch, args, arch_network_type, use_low_fidelity: bool = False):
    objs = utils.AverageMeter()
    n_diff_pairs_meter = utils.AverageMeter()
    all_data_loader_cycle = itertools.cycle(all_data_loader)
    model.train()

    # 每个微调轮次从两个筛选集合中分别抽取至多 semi_batch_size 对。
    n_pairs = min(len(high_quality_pair1), len(high_quality_pair2))
    if n_pairs == 0:
        raise RuntimeError("GMM 未筛选出可用于半监督微调的架构对。")
    n_max_pairs = min(1024, n_pairs)
    keep_inds1 = np.random.choice(
        np.arange(len(high_quality_pair1)), n_max_pairs, replace=False
    )
    keep_inds2 = np.random.choice(
        np.arange(len(high_quality_pair2)), n_max_pairs, replace=False
    )
    selected_pair1 = [high_quality_pair1[i] for i in keep_inds1]
    selected_pair2 = [high_quality_pair2[i] for i in keep_inds2]

    for step, (archs, real_accs, low_fidelity, index) in enumerate(real_loader):
        archs = np.array(archs)
        real_accs = np.array(real_accs)
        low_fidelity = np.array(low_fidelity)
        accs = low_fidelity if use_low_fidelity else real_accs
        n = len(archs)
        all_archs, _, all_low_fidelity, all_index = next(all_data_loader_cycle)
        if args.compare:
            archs_1, archs_2, better_lst, index_1, index_2 = compare_data_index(archs, index, accs, accs, args)
            n_diff_pairs = len(better_lst)
            n_diff_pairs_meter.update(float(n_diff_pairs))
            #loss = model.update_compare(archs_1, archs_2, better_lst)
            to_remove = {
                tuple(sorted((int(i), int(j))))
                for i, j in zip(index_1, index_2)
            }
            
            # 保留两个预测器各自筛选出的原始架构对，并排除当前真实架构对。
            index_11_12_cleaned = list({
                tuple(sorted((int(i), int(j)))) for i, j in selected_pair1
            } - to_remove)
            index_21_22_cleaned = list({
                tuple(sorted((int(i), int(j)))) for i, j in selected_pair2
            } - to_remove)
            if not index_11_12_cleaned or not index_21_22_cleaned:
                continue

          
            all_archs_11, all_archs_12, all_better_lst1, all_index_11, all_index_12 = batch_process_pairs(
                all_archs, 
                all_index, 
                all_low_fidelity, 
               index_11_12_cleaned,
               len(index_11_12_cleaned)
            )
            all_archs_21, all_archs_22, all_better_lst2, all_index_21, all_index_22 = batch_process_pairs(
               all_archs, 
                all_index, 
                all_low_fidelity,
               index_21_22_cleaned,
               len(index_21_22_cleaned)
            )
            all_better_lst1 = np.array(all_better_lst1).reshape(-1)
            all_better_lst2 = np.array(all_better_lst2).reshape(-1)
            n_diff_pairs = (
                len(better_lst) + len(all_better_lst1) + len(all_better_lst2)
            )
            n_diff_pairs_meter.update(float(n_diff_pairs))

            loss = model.update_compare(
                archs_1, archs_2, better_lst,
                all_archs_11[0], all_archs_12[0], all_better_lst1,
                all_archs_21[0], all_archs_22[0], all_better_lst2,
                args.finetune_mixup_weight, args.offset_weight)
            objs.update(loss, n_diff_pairs)
        else:
            n = len(archs)
            loss = model.update_predict(archs, accs)
            objs.update(loss, n)

        if step % args.report_freq == 0:
            n_pair_per_batch = (args.batch_size * (args.batch_size - 1)) // 2
            logging.info("train {:03d} [{:03d}/{:03d}] {:.4f}; {}".format(
                epoch, step, len(real_loader), objs.avg,
                "different pair ratio: {:.3f} ({:.1f}/{:3d})".format(
                    n_diff_pairs_meter.avg / n_pair_per_batch,
                    n_diff_pairs_meter.avg, n_pair_per_batch) if args.compare else ""))
    return objs.avg



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type = str, default='')
    parser.add_argument("--train-ratio", type = float, default = 0.01, 
            help = "Proportions of training samples")
    parser.add_argument("--pretrain-ratio", type = float, default = 1.0, 
            help = "Proportions of pre-training samples")
    parser.add_argument("--train-pkl", type = str, default='nasbench201_train.pkl', help = "Training Datasets pickle")
    parser.add_argument("--valid-pkl", type = str, default='nasbench201_valid.pkl', help = "Evaluate Datasets pickle")
    parser.add_argument("--no-pretrain", default = False, action = "store_true", 
            help = "Directly train the predictor without pretraining")
    parser.add_argument("--low-fidelity-type", type = str, default = "one_shot",
            help = "Applied low-fidelity information type")
    parser.add_argument("--train-dir", default = '', help = "Save train log / results into TRAIN_DIR")
    parser.add_argument("--seed", default = 2, type = int)
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--load", default = None, help = "Load comparator from disk.")
    parser.add_argument("--num-workers", default = 0, type = int)
    parser.add_argument("--test-every", default = 10, type = int)
    parser.add_argument("--test-only", default = False, action = "store_true")
    args = parser.parse_args()

    setproctitle.setproctitle("python {} config: {}; train_dir: {}; cwd: {}"\
                              .format(__file__, args.cfg_file, args.train_dir, os.getcwd()))

    # ---- Preparation & Setups ---- 

    # Set log
    log_format = "%(asctime)s %(message)s"
    logging.basicConfig(
        stream = sys.stdout, level = logging.INFO, 
        format = log_format, datefmt = "%m/%d %I:%M:%S %p"
    )
    logger = logging.getLogger()

    # Set train_dir
    if not args.test_only:
        assert args.train_dir is not None, "Must specificy `--train-dir` when training"
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)

        backup_cfg_file = os.path.join(args.train_dir, "config.yaml")
        shutil.copyfile(args.cfg_file, backup_cfg_file)
    else:
        backup_cfg_file = args.cfg_file

    # Device setup
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info("GPU device = {}".format(args.gpu))
        device = torch.device("cuda")
    else:
        logging.info("no GPU available, use CPU!!")
        device = torch.device("cpu")

    # Seed setup
    if args.seed is not None:
        utils.set_seed(args.seed)
        logging.info("Set seed: {}".format(args.seed))

    # Load and update the configuration
    with open(backup_cfg_file, "r") as cfg_f:
        cfg = yaml.load(cfg_f, Loader = yaml.FullLoader)

    cfg["train_ratio"] = args.train_ratio
    cfg["pretrain_ratio"] = args.pretrain_ratio
    cfg["low_fidelity_type"] = args.low_fidelity_type

    logging.info("Config: %s", cfg)

    args.__dict__.update(cfg)
    logging.info("Combined args: %s", args)

    # Data setup
    logging.info("Load pkl cache from {} and {}".format(args.train_pkl, args.valid_pkl))
    with open(args.train_pkl, "rb") as rf:
        train_data = pickle.load(rf)
    with open(args.valid_pkl, "rb") as rf:
        valid_data = pickle.load(rf)

    logging.info("Pretrain dataset ratio: %.3f; Train dataset ratio: %.3f", args.pretrain_ratio, args.train_ratio)
    _num = len(train_data)

    real_data = train_data[:int(_num * args.train_ratio)]
    un_data = train_data[int(_num * args.train_ratio): 3*int(_num * args.train_ratio)]
    real_data = Index_LFArchDataset(real_data, args.low_fidelity_type)
    train_data = train_data[:int(_num * args.pretrain_ratio)]
    train_data = Index_LFArchDataset(train_data, args.low_fidelity_type)
    valid_data = Index_LFArchDataset(valid_data, args.low_fidelity_type)
    un_data = Index_LFArchDataset(un_data, args.low_fidelity_type)
    total_data = train_data + valid_data

    logging.info("Number of architectures: pre-train: %d; train: %d; valid: %d", \
            len(train_data), len(real_data), len(valid_data))

    train_loader = DataLoader(
        train_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    val_loader = DataLoader(
        valid_data, batch_size = args.batch_size, shuffle = False, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    real_loader = DataLoader(
        real_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    total_loader = DataLoader(
        total_data, batch_size = args.batch_size, shuffle = False, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    
    un_train_loader = DataLoader(
        un_data, batch_size = args.batch_size, shuffle = True, pin_memory = True, 
        num_workers = args.num_workers, collate_fn = lambda items: list(zip(*items)))
    # ---- End of Preparation & Setups ----

    # ---- Predictor Initialization ----

    model = NoiseComparator(
        arch_embedder_cls = NasBench201_LSTMSeqEmbedder, # specify the architecture encoder type
        **cfg.pop("arch_network_cfg"),
        num_total_archs = len(train_data)
    )

    if args.load is not None:
        logging.info("Load checkpoint from {}".format(args.load))
        model.load(args.load)
    model.to(device)

    # ---- End of Predictor Initialization ----


    # ---- Initial Test ----
    #DELE报错low_fidelity_corr
    # if args.test_only:
    #     lf_corr, real_corr, patk = utils.valid(val_loader, model)
    #     logging.info("Valid: kendall tau {} {:.4f}; real {:.4f}; patk {}".format(
    #         args.low_fidelity_type, low_fidelity_corr, real_corr, patk))
    #     return
    
    if args.test_only:
        lf_corr, real_corr, patk = utils.valid(val_loader, model)
        logging.info("Valid: kendall tau {} {:.4f}; real {:.4f}; patk {}".format(
            args.low_fidelity_type, lf_corr, real_corr, patk))
        return

    # ---- End of Initial Test ----


    # ---- Training Step 1 ----
    # Pre-train the predictor with low_fidelity objective


    # if not args.no_pretrain:
    #     for i_epoch in range(1, args.epochs + 1):
    #         model.on_epoch_start(i_epoch)
    #         avg_loss = utils.noise_train_2(
    #             train_loader, 
    #             model, 
    #             i_epoch, 
    #             args.compare, 
    #             True, 
    #             args.max_compare_ratio,
    #             args.compare_threshold, 
    #             args.choose_pair_criterion
    #         )
    #         logging.info("Pre-train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

    #         if i_epoch == args.finetune_epochs or i_epoch % args.test_every == 0:
    #             lf_corr, real_corr, patk = utils.valid(
    #                 val_loader, model, os.path.join(args.train_dir, "stats.pkl"))
    #             logging.info(
    #                 "Pre-valid: Epoch {:3d}: kendall tau {} {:.4f}; real {:.4f}; patk {}".\
    #                 format(i_epoch, args.low_fidelity_type, lf_corr, real_corr, patk)
    #             )

    #     save_path = os.path.join(args.train_dir, "pre_train.ckpt")
    #     model.save(save_path)
    #     logging.info("Finish pre-training. Save checkpoint to {}".format(save_path))

    # if not args.no_pretrain:
    #     for i_epoch in range(1, 101):
    #         model.on_epoch_start(i_epoch)
           
    #         avg_loss = utils.noise_train(
    #             train_loader, 
    #             val_loader, 
    #             model, 
    #             i_epoch, 
    #             args.compare, 
    #             True, 
    #             args.max_compare_ratio,
    #             args.compare_threshold, 
    #             args.choose_pair_criterion
    #         )
            
    #         logging.info("Pre-train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

    #         if i_epoch == args.finetune_epochs or i_epoch % args.test_every == 0:
    #             lf_corr, real_corr, patk = utils.valid(
    #                 val_loader, model, os.path.join(args.train_dir, "stats.pkl"))
    #             logging.info(
    #                 "Pre-valid: Epoch {:3d}: kendall tau {} {:.4f}; real {:.4f}; patk {}".\
    #                 format(i_epoch, args.low_fidelity_type, lf_corr, real_corr, patk)
    #             )

    #     save_path = os.path.join(args.train_dir, "pre_train.ckpt")
    #     model.save(save_path)
    #     logging.info("Finish pre-training. Save checkpoint to {}".format(save_path))


    if not args.no_pretrain:
        for i_epoch in range(1, args.epochs + 1):
            model.on_epoch_start(i_epoch)
           
            avg_loss = utils.noise_train(
                train_loader, 
                val_loader, 
                model, 
                i_epoch, 
                args.compare, 
                True, 
                args.max_compare_ratio,
                args.compare_threshold, 
                args.choose_pair_criterion
            )
        #     avg_loss = utils.train_m(
        #         train_loader, 
        #         model, 
        #         i_epoch, 
        #         args.compare, 
        #         True, 
        #         args.max_compare_ratio,
        #         args.compare_threshold, 
        #         args.choose_pair_criterion
        # )
            logging.info("Pre-train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

            if i_epoch == args.finetune_epochs or i_epoch % args.test_every == 0:
                lf_corr, real_corr, patk = utils.index_valid(
                    val_loader, model, os.path.join(args.train_dir, "stats.pkl"))
                logging.info(
                    "Pre-valid: Epoch {:3d}: kendall tau {} {:.4f}; real {:.4f}; patk {}".\
                    format(i_epoch, args.low_fidelity_type, lf_corr, real_corr, patk)
                )

        save_path = os.path.join(args.train_dir, "pre_train.ckpt")
        model.save(save_path)
        logging.info("Finish pre-training. Save checkpoint to {}".format(save_path))

    # ---- End of Training Step 1 ----


    # ---- Training Step 2 ----
    # Finetune the predictor with the actual performance

    model.reinit_optimizer()
    model.reinit_scheduler()

    for i_epoch in range(1, args.finetune_epochs + 1):
        model.on_epoch_start(i_epoch)
        avg_loss = utils.semi_train(
                real_loader, 
                un_train_loader,
                model, 
                i_epoch, 
                args.compare, 
                False, 
                args.max_compare_ratio,
                args.compare_threshold, 
                args.choose_pair_criterion
        )
        # avg_loss = utils.train_m(
        #         real_loader, 
        #         model, 
        #         i_epoch, 
        #         args.compare, 
        #         False, 
        #         args.max_compare_ratio,
        #         args.compare_threshold, 
        #         args.choose_pair_criterion
        # )
        logging.info("Train: Epoch {:3d}: train loss {:.4f}".format(i_epoch, avg_loss))

        if i_epoch == args.finetune_epochs or i_epoch % args.test_every == 0:
            lf_corr, real_corr, patk = utils.index_valid(
                val_loader, model, os.path.join(args.train_dir, "stats.pkl")
            )
            logging.info(
                "Valid: Epoch {:3d}: kendall tau {} {:.4f}; real {:.4f}; patk {}".\
                format(i_epoch, args.low_fidelity_type, lf_corr, real_corr, patk)
            )

    save_path = os.path.join(args.train_dir, "final.ckpt")
    model.save(save_path)
    logging.info("Save checkpoint to {}".format(save_path))

    # ---- End of Training Step 2 ----


if __name__ == "__main__":
    main()
