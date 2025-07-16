import time
import torch
import torch.nn as nn
import argparse
import os
import numpy as np
import pickle
from utils.utils import fix_random_seed
from utils.dataset import get_dataset
from utils.models import get_model
from utils.metrics import cal_metric
from torch.utils.data import DataLoader
from utils.get_stat import get_features
def get_eval_options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ind_dataset", type=str, default="ImageNet",help="in-distribution dataset name")
    parser.add_argument("--ood_dataset",type=str,nargs='+', default=["iNat", "SUN","Places","Textures", "openimage_o","imagenet_o"],help="OOD dataset list")    
    parser.add_argument("--model", type=str, default="resnet",choices=["resnet","vit","convnext","densenet","regnet","efficientnet","swin"],help="model name")
    parser.add_argument("--gpu", type=int, default=1,help="gpu id")
    parser.add_argument('--num_classes', type=int, default=1000,help="number of classes")
    parser.add_argument("--random_seed", type=int, default=0,help="random seed")
    parser.add_argument("--bs", type=int, default=32,help="batch size")
    parser.add_argument("--OOD_method", type=str, default="CADRef",help="OOD method name",choices=["MSP","ODIN","Energy","GEN","ReAct","DICE","GradNorm","MaxLogit","ASH","OptFS","VIM","Residual","CARef","CADRef"])
    parser.add_argument("--use_feature_cache", type=bool, default=True, help="use feature cache")
    parser.add_argument("--use_score_cache", type=bool, default=True, help="use score cache")
    parser.add_argument("--cache_dir", type=str, default="cache",help="cache directory")
    parser.add_argument("--result_dir", type=str, default="result",help="result directory")
    parser.add_argument("--num_workers", type=int, default=24,help="number of workers")
    parser.add_argument("--logit_method", type=str, default="Energy",choices=["Energy","MSP","MaxLogit","GEN"],help="logit method for CADRef")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_eval_options()
    fix_random_seed(args.random_seed)
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    


    for dir_path in [
        args.cache_dir,
        args.result_dir,
        os.path.join(args.cache_dir, args.model),
        os.path.join(args.result_dir, args.model),
        os.path.join(args.cache_dir, args.model, args.ind_dataset),
        os.path.join(args.result_dir, args.model, args.ind_dataset)
    ]:
        os.makedirs(dir_path, exist_ok=True)

    if args.OOD_method in ['GradNorm']:
        args.bs = 1

    if args.ind_dataset == "ImageNet":
        args.num_classes = 1000
    elif args.ind_dataset == "cifar10":
        args.num_classes = 10
    elif args.ind_dataset == "cifar100":
        args.num_classes = 100
    print(args)
    


    model = get_model(args).to(device)
    model.eval()
    
    ind_scores, ood_scores = None, None

    if args.OOD_method == "MSP":
        from ood_methods.MSP import MSP
        evaluator = MSP(model, args, device)


    elif args.OOD_method == "ODIN":
        from ood_methods.ODIN import ODIN
        evaluator = ODIN(model, args, device)

    elif args.OOD_method == "Energy":
        from ood_methods.Energy import Energy
        evaluator = Energy(model, args, device)

    elif args.OOD_method == "GEN":
        from ood_methods.GEN import GEN
        evaluator = GEN(model, args, device)

    elif args.OOD_method == "ReAct":
        from ood_methods.ReAct import ReAct
        evaluator = ReAct(model, args, device)

        threshold_file_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, "ReAct_threshold.pkl")
        if os.path.exists(threshold_file_path) and args.use_feature_cache:
            with open(threshold_file_path, "rb") as f:
                threshold = pickle.load(f)
        else:
            train_data, _ = get_dataset(args.ind_dataset,args)
            train_loader = DataLoader(dataset=train_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
            features, logits = get_features(model, train_loader, args, device)
            threshold = evaluator.get_threshold(features)
            with open(threshold_file_path, "wb") as f:
                pickle.dump(threshold, f)
        
        evaluator.set_state(threshold)


    elif args.OOD_method == "DICE":
        from ood_methods.DICE import DICE
        evaluator = DICE(model, args, device)

        mask_file_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, "DICE_mask.pkl")
        if os.path.exists(mask_file_path) and args.use_feature_cache:
            with open(mask_file_path, "rb") as f:
                mask = pickle.load(f)
        else:
            train_data, _ = get_dataset(args.ind_dataset,args)
            train_loader = DataLoader(dataset=train_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
            features, logits = get_features(model, train_loader, args, device)
            mask = evaluator.get_mask(features)
            with open(mask_file_path, "wb") as f:
                pickle.dump(mask, f)
        
        evaluator.set_state(mask)
    elif args.OOD_method == "GradNorm":
        from ood_methods.GradNorm import GradNorm
        evaluator = GradNorm(model, args, device)

    elif args.OOD_method == "MaxLogit":
        from ood_methods.MaxLogit import MaxLogit
        evaluator = MaxLogit(model, args, device)

    elif args.OOD_method == "ASH":
        from ood_methods.ASH import ASH
        evaluator = ASH(model,args, device)

    elif args.OOD_method == "OptFS":
        from ood_methods.OptFS import OptFS
        evaluator = OptFS(model, args, device)
        
        shaping_parameter_file_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, "OptFS_shaping_parameter.pkl")
        if os.path.exists(shaping_parameter_file_path) and args.use_feature_cache:
            with open(shaping_parameter_file_path, "rb") as f:
                theta, left_boundary, width = pickle.load(f)
        else:
            train_data, _ = get_dataset(args.ind_dataset,args)
            train_loader = DataLoader(dataset=train_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
            features, logits = get_features(model, train_loader, args, device)
            theta, left_boundary, width = evaluator.get_optimal_shaping(features, logits)
            with open(shaping_parameter_file_path, "wb") as f:
                pickle.dump((theta, left_boundary, width), f)
        
        evaluator.set_state(theta, left_boundary, width)

    elif args.OOD_method == "VIM":
        from ood_methods.VIM import VIM
        evaluator = VIM(model, args, device)

        NS_file_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, "VIM_NS.pkl")
        if os.path.exists(NS_file_path) and args.use_feature_cache:
            with open(NS_file_path, "rb") as f:
                NS, alpha,u = pickle.load(f)
        else:
            train_data, _ = get_dataset(args.ind_dataset,args)
            train_loader = DataLoader(dataset=train_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
            features, logits = get_features(model, train_loader, args, device)
            NS, alpha,u = evaluator.get_ns(features)
            with open(NS_file_path, "wb") as f:
                pickle.dump((NS, alpha,u), f)
        
        evaluator.set_state(NS, alpha,u)
    elif args.OOD_method == "Residual":
        from ood_methods.VIM import VIM
        evaluator = VIM(model, args, device, mode="Residual")

        NS_file_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, "VIM_NS.pkl")
        if os.path.exists(NS_file_path) and args.use_feature_cache:
            with open(NS_file_path, "rb") as f:
                NS, alpha,u = pickle.load(f)
        else:
            train_data, _ = get_dataset(args.ind_dataset,args)
            train_loader = DataLoader(dataset=train_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
            features, logits = get_features(model, train_loader, args, device)
            NS, alpha,u = evaluator.get_ns(features)
            with open(NS_file_path, "wb") as f:
                pickle.dump((NS, alpha,u), f)
        
        evaluator.set_state(NS, alpha,u)
    
    elif args.OOD_method == "CARef":
        from ood_methods.CARef import CARef
        evaluator = CARef(model, args, device)
        caref_file_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, "CARef_feature_mean.pkl")
        if os.path.exists(caref_file_path) and args.use_feature_cache:
            with open(caref_file_path, "rb") as f:
                feature_mean = pickle.load(f)
        else:
            train_data, _ = get_dataset(args.ind_dataset,args)
            train_loader = DataLoader(dataset=train_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
            features, logits = get_features(model, train_loader, args, device)
            feature_mean = evaluator.get_mean_feature(features)
            with open(caref_file_path, "wb") as f:
                pickle.dump(feature_mean, f)
        
        evaluator.set_state(feature_mean)
    elif args.OOD_method == "CADRef":
        from ood_methods.CADRef import CADRef
        evaluator = CADRef(model, args, device)
        cadref_file_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, "CADRef_"+args.logit_method+".pkl")
        if os.path.exists(cadref_file_path) and args.use_feature_cache:
            with open(cadref_file_path, "rb") as f:
                train_mean, global_mean_logit_score = pickle.load(f)
        else:
            train_data, _ = get_dataset(args.ind_dataset,args)
            train_loader = DataLoader(dataset=train_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
            features, logits = get_features(model, train_loader, args, device)
            train_mean, global_mean_logit_score = evaluator.get_state(features, logits)
            with open(cadref_file_path, "wb") as f:
                pickle.dump((train_mean, global_mean_logit_score), f)
        
        evaluator.set_state(train_mean, global_mean_logit_score)

    ind_score_cache_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, args.OOD_method+"_ind_scores.pkl")
    if args.OOD_method == "CADRef":
        ind_score_cache_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, args.OOD_method+"_"+args.logit_method+"_ind_scores.pkl")
    if os.path.exists(ind_score_cache_path) and args.use_score_cache:
        with open(ind_score_cache_path, "rb") as f:
            ind_scores = pickle.load(f)
    else:
        _, ind_data = get_dataset(args.ind_dataset,args)
        ind_loader = DataLoader(dataset=ind_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
        ind_scores = evaluator.eval(ind_loader)
        with open(ind_score_cache_path, "wb") as f:
            pickle.dump(ind_scores, f)
    ind_labels = np.ones(ind_scores.shape[0])
    for ood_dataset in args.ood_dataset:
        ood_score_cache_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, args.OOD_method+"_"+ood_dataset+"_scores.pkl")
        if args.OOD_method == "CADRef":
            ood_score_cache_path = os.path.join(args.cache_dir, args.model, args.ind_dataset, args.OOD_method+"_"+args.logit_method+"_"+ood_dataset+"_scores.pkl")
        if os.path.exists(ood_score_cache_path) and args.use_score_cache:
            with open(ood_score_cache_path, "rb") as f:
                ood_scores = pickle.load(f)
        else:
            _, ood_data = get_dataset(ood_dataset,args)
            ood_loader = DataLoader(dataset=ood_data, batch_size=args.bs, pin_memory=True, num_workers=args.num_workers, shuffle=False)
            ood_scores = evaluator.eval(ood_loader)
            with open(ood_score_cache_path, "wb") as f:
                pickle.dump(ood_scores, f)
        ood_labels = np.zeros(ood_scores.shape[0])
        scores = np.concatenate([ind_scores, ood_scores])
        labels = np.concatenate([ind_labels, ood_labels])
        auroc, aupr, fpr = cal_metric(labels, scores)
        auroc = round(auroc*100, 4)
        fpr = round(fpr*100, 4)
        print("{:10} {:10} {:10} {:10}".format(args.OOD_method, ood_dataset, auroc, fpr))
        with open(os.path.join(args.result_dir, args.model,args.ind_dataset, args.OOD_method+".txt"), "a") as f:
            f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), ood_dataset, auroc, fpr))
      

