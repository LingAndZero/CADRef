import torch



def get_model(args):
    model = None

    if args.model == "resnet":
        import models.imagenet.resnet as resnet
        model = resnet.resnet50(num_classes=args.num_classes, pretrained=True)

    elif args.model == "densenet":
        if args.ind_dataset in ['cifar10', 'cifar100']:
            import models.cifar.densenet as densenet
            model = densenet.DenseNet3(depth=100, reduction=0.5, bottleneck=True, num_classes=args.num_classes)
            filename = "checkpoints/" + args.ind_dataset + "-" + args.model + "/last.pth.tar"
            model.load_state_dict((torch.load(filename)['state_dict']))
        else:
            import models.imagenet.densenet as densenet
            model = densenet.densenet201(weights='IMAGENET1K_V1')

    elif args.model == "vit":
        import models.imagenet.vision_transformer as vit
        model = vit.vit_b_16(weights='IMAGENET1K_V1')

    elif args.model == "swin":
        import models.imagenet.swin_transformer as swin
        model = swin.swin_b(weights='IMAGENET1K_V1')

    elif args.model == "convnext":
        import models.imagenet.convnext as convnext
        model = convnext.convnext_base(weights='IMAGENET1K_V1')

    elif args.model == "regnet":
        import models.imagenet.regnet as regnet
        model = regnet.regnet_x_8gf(weights='IMAGENET1K_V1')

    elif args.model == "efficientnet":
        import models.imagenet.efficientnet as efficientnet
        model = efficientnet.efficientnet_v2_m(weights='IMAGENET1K_V1')
    
    elif args.model == "maxvit":
        import models.imagenet.maxvit as maxvit
        model = maxvit.maxvit_t(weights='IMAGENET1K_V1')

    return model


