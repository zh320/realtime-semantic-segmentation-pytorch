from torchmetrics import JaccardIndex


def get_seg_metrics(config, task='multiclass', reduction='none'):
    metrics = JaccardIndex(task=task, num_classes=config.num_class, 
                            ignore_index=config.ignore_index, average=reduction,)
    return metrics