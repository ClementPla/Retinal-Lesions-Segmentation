import segmentation_models_pytorch as smp
from nntools.nnet.models import R2UNet, R2AttUNet, AttUNet, UNet, MultiTaskUnet, NestedUNet
from .contrastive_unet import ContrastiveUnet
from .multi_dataset_unet import MultiDatasetUnet

def get_network(config):
    if config['architecture'] == 'Contrastive_Unet':
        network = ContrastiveUnet(classes=config['n_classes'],
                                  encoder_weights='imagenet' if config['pretrained'] else None)

    if config['architecture'] == 'Original_Unet':
        network = UNet(output_ch=config['n_classes'])

    if config['architecture'] == 'MultiDatasetUnet':
        network = MultiDatasetUnet(datasets=config['datasets'], classes=config['n_classes'],
                                   encoder_weights='imagenet' if config['pretrained'] else None)

    if config['architecture'] == 'Unet-resnet101':
        network = smp.Unet('resnet101', classes=config['n_classes'],
                           encoder_weights='imagenet' if config['pretrained'] else None)
    elif config['architecture'] == 'Unet':
        network = smp.Unet(classes=config['n_classes'],
                           encoder_weights='imagenet' if config['pretrained'] else None)
    elif config['architecture'] == 'DeepLabV3+':
        network = smp.DeepLabV3Plus('resnet101', classes=config['n_classes'],
                                    encoder_weights='imagenet' if config['pretrained'] else None)

    elif config['architecture'] == 'PSPNet':
        network = smp.PSPNet('resnet101', classes=config['n_classes'],
                             encoder_weights='imagenet' if config['pretrained'] else None)

    elif config['architecture'] == 'R2UNet':
        network = R2UNet(output_ch=config['n_classes'])

    elif config['architecture'] == 'R2AttUNet':
        network = R2AttUNet(output_ch=config['n_classes'])

    elif config['architecture'] == 'AttUNet':
        network = AttUNet(output_ch=config['n_classes'])

    elif config['architecture'] == 'MultiTaskUnet':
        network = MultiTaskUnet(output_chs=(6, 6))

    elif config['architecture'] == 'NestedUnet':
        network = NestedUNet(output_ch=config['n_classes'])

    elif config['architecture'] == 'HRNet':
        from networks.hrnet import HRN
        network = HRN(num_classes=config['n_classes'])

    elif config['architecture'] == 'DeepLabV3+':
        network = smp.DeepLabV3Plus('resnet101', classes=config['n_classes'],
                                    encoder_weights='imagenet' if config['pretrained'] else None)
    return network

