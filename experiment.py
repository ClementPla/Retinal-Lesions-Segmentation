import os

import albumentations as A
import cv2
import nntools.tracker.metrics as NNmetrics
import numpy as np
import tqdm
import math
from nntools.experiment import SupervisedExperiment
from nntools.utils import reduce_tensor
from torchmetrics import AUC, BinnedPrecisionRecallCurve
import torch.nn.functional as F
from constants import LESIONS_LABELS
from networks import get_network
from tools.losses import *
from retinal_dataset.datasets_setup import add_operations_to_dataset, get_datasets_from_config
from tools.contrastive_learning import ContrastiveLoss, contrastive_loss_from_batch
from tools.utils import DA, get_augmentation_functions, Dataset, Lesions
import ttach as tta
import torch

class RetinExp(SupervisedExperiment):
    def __init__(self, config, run_id=None,
                 train_sets=Dataset.IDRID | Dataset.MESSIDOR | Dataset.FGADR,
                 test_sets=Dataset.IDRID | Dataset.RETINAL_LESIONS,
                 da_level=DA.COLOR | DA.GEOMETRIC,
                 labels=Lesions.EXUDATES | Lesions.MICROANEURYSMS | Lesions.COTTON_WOOL_SPOT | Lesions.HEMORRHAGES,
                 save_prediction=False, cache=False,
                 contrastive_loss=ContrastiveLoss.SINGLE_IMAGE):

        if labels.length == 4 and config['Network']['n_classes'] == labels.length+1:
            self.multi_class = True
        else:
            self.multi_class = False
            config['Network']['n_classes'] = labels.length

        super(RetinExp, self).__init__(config, run_id)
        self.multiDatasetRegularization = self.c['Network']['architecture'] == 'MultiDatasetUnet'
        if self.multiDatasetRegularization:
            self.c['Network']['datasets'] = train_sets
        self.set_model(get_network(self.c['Network']))

        d = get_datasets_from_config(self.c['Dataset'], sets=train_sets, seed=self.seed,
                                     split_ratio=self.c['Validation']['size'], labels=labels)
        # Return a dict with two keys (core and split)

        aug_func = get_augmentation_functions(da_level)
        aug = A.Compose(aug_func)
        ops = [aug, A.Normalize(mean=self.c['Preprocessing']['mean'],
                                std=self.c['Preprocessing']['std'],
                                always_apply=True)]

        if self.c['Preprocessing']['random_crop']:
            ops.append(A.CropNonEmptyMaskIfExists(*self.c['Preprocessing']['crop_size'], always_apply=True))

        add_operations_to_dataset(d['core'], ops)
        add_operations_to_dataset(d['split'], A.Normalize(mean=self.c['Preprocessing']['mean'],
                                                          std=self.c['Preprocessing']['std'],
                                                          always_apply=True))
        if cache:
            for dataset in d['core']:
                dataset.cache()
        self.set_train_dataset(d['core'])
        self.set_valid_dataset(d['split'])

        if self.multiDatasetRegularization:
            self.data_keys = ['image', 'Dataset']
            self.train_dataset.return_tag = True
            self.validation_dataset.return_tag = True

        augmentation_factor = self.c['Dataset'].get('augmentation_factor', None)
        if augmentation_factor:
            self.train_dataset.multiply_size(augmentation_factor)

        """
        Configure the test set
        """
        d = get_datasets_from_config(self.c['Test'], sets=test_sets,
                                     seed=self.seed, split_ratio=0,
                                     shape=self.c['Dataset']['shape'], labels=labels)
        add_operations_to_dataset(d['core'], A.Normalize(mean=self.c['Preprocessing']['mean'],
                                                         std=self.c['Preprocessing']['std'],
                                                         always_apply=True))
        self.set_test_dataset(d['core'])
        """
        Define optimizers
        """
        self.set_optimizer(**self.c['Optimizer'])
        self.set_scheduler(**self.c['Learning_rate_scheduler'])
        self.model.pr_curve = BinnedPrecisionRecallCurve(num_classes=self.n_classes, num_thresholds=33)
        self.auc = AUC(reorder=True)
        print("Training size %i, validation size %i, test size %i" % (len(self.train_dataset),
                                                                      len(self.validation_dataset),
                                                                      sum([len(d) for d in self.test_dataset])))
        self.log_artifacts(os.path.realpath(__file__))
        self.log_params(DA=da_level.name)
        self.save_prediction = save_prediction

        self.contrastive_pretrain = self.c['Training'].get('contrastive_pretraining', False)
        self.only_inference = False

        if self.contrastive_pretrain:
            self.contrastive_loss = contrastive_loss
            self.log_params(Contrastive_loss=contrastive_loss.name)

        self.tta_transforms = tta.Compose([tta.HorizontalFlip(),
                                           tta.Rotate90(angles=[0, 90, 180, 270]),
                                           tta.Scale(scales=[1, 1.25, 0.75, 0.5]),
                                           ])

    def validate(self, model, valid_loader, iteration, rank=0, loss_function=None):
        if self.contrastive_pretrain:
            return None
        model.eval()
        gpu = self.get_gpu_from_rank(rank)
        if self.multi_class:
            confMat = torch.zeros(self.n_classes, self.n_classes)
        else:
            confMat = torch.zeros(self.n_classes, 2, 2)
        confMat = confMat.cuda(gpu)
        losses = []
        for n, batch in enumerate(valid_loader):
            batch = self.batch_to_device(batch, rank)
            gt = batch['mask']
            proba = model(*self.pass_data_keys_to_model(batch))
            losses.append(torch.nan_to_num(loss_function(*proba, y_true=gt)).item())
            if self.multiDatasetRegularization:
                proba = proba[0]
            if not self.multi_class:
                preds = torch.sigmoid(proba) >= 0.5
            else:
                max_arg = torch.max(gt, 1, keepdim=False)
                gt = max_arg.indices + 1
                gt[max_arg.values == 0] = 0
                preds = torch.argmax(torch.softmax(proba, 1), 1)

            confMat += NNmetrics.confusion_matrix(preds, gt, num_classes=self.n_classes, multilabel=not self.multi_class)

        if self.multi_gpu:
            confMat = reduce_tensor(confMat, self.world_size, mode='sum')
        confMat = NNmetrics.filter_index_cm(confMat, self.ignore_index)
        mIoU = NNmetrics.mIoU_cm(confMat)
        if self.is_main_process(rank):
            stats = NNmetrics.report_cm(confMat)
            stats['mIoU'] = mIoU
            stats['Validation loss'] = np.mean(losses)
            self.log_metrics(step=iteration, **stats)
            if self.tracked_metric is None or mIoU >= self.tracked_metric:
                self.tracked_metric = mIoU
                filename = ('best_valid_iteration_%i_mIoU_%.3f' % (iteration, mIoU)).replace('.', '')
                self.save_model(model, filename=filename)

        model.train()
        return mIoU

    def contrastive_training(self, model, batch, rank):
        iteration = self.ctx_train['iteration']
        if iteration <= 1:
            for g in self.ctx_train['optimizer'].param_groups:
                g['lr'] = self.c['Contrastive_training']['lr']

        if iteration > self.c['Contrastive_training']['training_step']:
            # Ends the contrastive pretraining and jumps to the regular forward_train function
            self.contrastive_pretrain = False
            optimizer = self.partial_optimizer(
                model.get_trainable_parameters(self.c['Optimizer']['params_solver']['lr']))
            self.ctx_train['optimizer'] = optimizer
            lr_scheduler = self.partial_lr_scheduler(self.ctx_train['optimizer'])
            self.ctx_train['lr_scheduler'] = lr_scheduler
            model.activate_segmentation_mode()
        else:
            return contrastive_loss_from_batch(model, batch['image'], batch[self.gt_name],
                                               loss_type=self.contrastive_loss,
                                               **self.c['Contrastive_training'])

    def multidataset_training(self, model, loss_function, batch, rank):
        iteration = self.ctx_train['iteration']
        if (iteration == 0) or (iteration ==1):
            model.grads_control(encoder=False, decoder=False, segmentation_head=False, cm_predictors=True)
        if iteration <= self.c['Training']['MultiDataset']['trace_warm_up']:
            y, cm_predictions, tag = model(*self.pass_data_keys_to_model(batch))
            loss = 0
            for tag, c in cm_predictions:
                b, d, h, w = c.shape
                n_classes = int(math.sqrt(d))
                c = c.view(b, d, h * w).permute(0, 2, 1)
                c = c.reshape(b * h * w, d).view(b * h * w, n_classes, n_classes)
                c = c / (c.sum(1, keepdim=True) + 0.001)
                target = torch.eye(n_classes).view(1, n_classes, n_classes).to(c)
                target = target.repeat(b * h * w, 1, 1)
                loss += F.mse_loss(c, target)
                # loss += - torch.trace(torch.sum(c, dim=0))/(b*h*w)

            if iteration == self.c['Training']['MultiDataset']['trace_warm_up']:
                self.save_model(model, filename='warm_up_training')
                model.grads_control(encoder=True, decoder=True, segmentation_head=True, cm_predictors=True)
                with torch.no_grad():
                    self.validate(model, self.ctx_train['valid_loader'], iteration, rank, loss_function)
            return loss
        else:
            return super(RetinExp, self).forward_train(model, loss_function, batch, rank)

    def forward_train(self, model, loss_function, batch, rank):
        if self.contrastive_pretrain:
            return self.contrastive_training(model, batch, rank)

        if self.multiDatasetRegularization:
            return self.multidataset_training(model, loss_function, batch, rank)

        if not self.contrastive_pretrain:
            return super(RetinExp, self).forward_train(model, loss_function, batch, rank)

    def end(self, model, rank):
        """
        This function must not be called by the user. It will be called automatically when the training ends or
        when this.eval() is called.
        :param model:
        :param rank:
        :return:
        """
        gpu = self.get_gpu_from_rank(rank)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
        model.load(self.tracker.network_savepoint,
                   load_most_recent=True,
                   map_location=map_location,
                   strict=False,
                   filtername='best_valid')
        model.eval()
        tta_model = tta.SegmentationTTAWrapper(model, self.tta_transforms, merge_mode='mean')

        for dataset in self.test_dataset:
            self.auc.reset()
            tta_model.model.pr_curve.reset()
            if self.is_main_process(rank):
                print("Testing on %s" % dataset.id)
            dataset.return_indices = True
            if self.only_inference:
                self.inference(tta_model, rank, dataset)
            else:
                self.inference_n_eval(tta_model, rank, dataset)

    def inference_n_eval(self, tta_model, rank, dataset):
        gpu = self.get_gpu_from_rank(rank)
        test_loader, test_sampler = self.get_dataloader(dataset, shuffle=False, batch_size=1,
                                                        persistent_workers=False,
                                                        rank=rank)

        suffix = dataset.id.suffix

        if self.multi_class:
            confMat = torch.zeros(self.n_classes, self.n_classes)
        else:
            confMat = torch.zeros(self.n_classes, 2, 2)
        confMat = confMat.cuda(gpu)
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                img = batch['image'].cuda(gpu)
                gt = batch['mask'].cuda(gpu)
                probas = tta_model(img)
                if self.multi_class:
                    max_arg = torch.max(gt, 1, keepdim=False)
                    gt = max_arg.indices + 1
                    gt[max_arg.values == 0] = 0
                    probas = torch.softmax(probas, 1)
                    preds = torch.argmax(probas, 1)

                else:
                    probas = torch.sigmoid(probas)
                    preds = probas > 0.5

                if self.save_prediction:
                    self.save_predictions(probas, batch)

                if self.multi_class:
                    tta_model.model.pr_curve.update(probas.permute((0, 2, 3, 1)).reshape(-1, self.n_classes), gt.flatten())
                else:
                    tta_model.model.pr_curve.update(probas.permute((0, 2, 3, 1)).reshape(-1, self.n_classes),
                                          gt.permute((0, 2, 3, 1)).reshape(-1, self.n_classes))
                confMat += NNmetrics.confusion_matrix(preds, gt, num_classes=self.n_classes, multilabel=not self.multi_class)
            if self.multi_gpu:
                confMat = reduce_tensor(confMat, self.world_size, mode='sum')
            precision, recall, thresholds = tta_model.model.pr_curve.compute()

        if self.is_main_process(rank):
            mIoU = NNmetrics.mIoU_cm(confMat)
            # stats = NNmetrics.report_cm(confMat)
            stats = {'mIoU': mIoU}
            test_scores = {}
            labels = []
            if self.multi_class:
                labels.append('Background')
            labels += LESIONS_LABELS
            for l, p, r in zip(labels, precision, recall):
                test_scores[l + '_roc' + suffix] = self.auc(r, p).item()
            for k in stats:
                test_scores['Test_%s' % k + suffix] = stats[k]
            self.log_metrics(step=0, **test_scores)
            confMat = confMat.cpu().numpy()
            np.save(os.path.join(self.tracker.prediction_savepoint, 'confMat' + suffix + '.npy'), confMat)
            self.log_artifacts(os.path.join(self.tracker.prediction_savepoint, 'confMat' + suffix + '.npy'))

    def inference(self, tta_model, rank, dataset):
        gpu = self.get_gpu_from_rank(rank)
        test_loader, test_sampler = self.get_dataloader(dataset, shuffle=False, batch_size=1,
                                                        persistent_workers=False,
                                                        rank=rank)
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
                img = batch['image'].cuda(gpu)
                probas = tta_model(img)
                probas = torch.sigmoid(probas)
                self.save_predictions(probas, batch)

    def save_predictions(self, predicted_probas, batch):
        indices = batch['index']
        filenames = self.test_dataset.filename(indices)
        if self.multi_class:
            predicted_probas = predicted_probas[:, 1:]  # We skip the first one (corresponding to the background class)
        for ps, f in zip(predicted_probas, filenames):
            for i, p in enumerate(ps):
                folder = os.path.join(self.tracker.prediction_savepoint, LESIONS_LABELS[i])
                if not os.path.exists(folder):
                    os.makedirs(folder)
                out = (p.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(folder, f), out)

