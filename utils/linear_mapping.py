def get_linear_layer_mapping(model_name, dataset, model):
    if model_name == "convnext":
        return model.classifier[-1]
    elif model_name == "convnext_small":
        return model.classifier[-1]
    elif model_name == "convnext_large":
        return model.classifier[-1]
    elif model_name == "vit" and dataset == "ImageNet":
        return model.heads[-1]
    elif model_name == "vit" and dataset in ["cifar10", "cifar100"]:
        return model.head
    elif model_name == "swin" or model_name == "swinv2":
        return model.head
    elif model_name == "densenet" and dataset in ['ImageNet']:
        return model.classifier
    elif model_name == "densenet" and dataset in ['cifar10', 'cifar100']:
        return model.fc
    elif model_name == "regnet":
        return model.fc
    elif model_name == "efficientnet":
        return model.classifier[-1]
    elif model_name == "efficientnet_b7":
        return model.classifier[-1]
    elif model_name == "resnet" and dataset in ['cifar10', 'cifar100']:
        return model.linear
    elif model_name == "resnet" and dataset == "ImageNet":
        return model.fc
    elif model_name == "vgg" and dataset in ['cifar10', 'cifar100']:
        return model.fc
    elif model_name == "mobilenet" and dataset in ['cifar10', 'cifar100']:
        return model.linear
    elif model_name == "maxvit":
        return model.classifier[-1]
    else:
        raise ValueError(f"Unsupported model: {model_name} for dataset: {dataset}")
