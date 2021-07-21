import os

import albumentations as A
import cv2
import nntools.tracker.metrics as NNmetrics
import numpy as np
import torch
import tqdm
from nntools.experiment import SupervisedExperiment
from nntools.utils import reduce_tensor
from torchmetrics import AUC, BinnedPrecisionRecallCurve

from constants import LESIONS_LABELS
from networks import get_network
from retinal_dataset.datasets_setup import add_operations_to_dataset, get_datasets_from_config
from scripts.utils import DA, DA_FUNCTIONS, Dataset


class RetinExp(SupervisedExperiment):
    def __init__(self, config, id=None,
                 train_sets=Dataset.IDRID | Dataset.MESSIDOR | Dataset.FGADR,
                 test_sets=Dataset.IDRID | Dataset.RETINAL_LESIONS,
                 DA_level=(DA.COLOR | DA.GEOMETRIC),
                 save_prediction=False):

        super(RetinExp, self).__init__(config, id)
        self.set_model(get_network(config['Network']))
        d = get_datasets_from_config(self.c['Dataset'], sets=train_sets, seed=self.seed,
                                     split_ratio=self.c['Validation']['size'])
        # Return a dict with two keys (core and split)

        aug_func = DA_FUNCTIONS[DA_level]
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
        self.set_train_dataset(d['core'])
        self.set_valid_dataset(d['split'])

        """
        Configure the test set
        """
        d = get_datasets_from_config(self.c['Test'], sets=test_sets,
                                     seed=self.seed, split_ratio=0,
                                     shape=self.c['Dataset']['shape'])
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
                                                                      len(self.test_dataset)))
        self.log_artifacts(os.path.realpath(__file__))
        self.log_params(DA=DA_level.name)
        self.save_prediction = save_prediction
        self.contrastive_pretrain = self.c['Training'].get('contrastive_pretraining', False)
        self.only_inference = False

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
        for dataset in self.test_dataset:
            self.auc.reset()
            model.pr_curve.reset()
            if self.is_main_process(rank):
                print("Testing on %s" % dataset.tag)
            dataset.return_indices = True
            if self.only_inference:
                self.inference(model, rank, dataset)
            else:
                self.inference_n_eval(model, rank, dataset)

    def inference_n_eval(self, model, rank, dataset):
        gpu = self.get_gpu_from_rank(rank)
        test_loader, test_sampler = self.get_dataloader(dataset, shuffle=False, batch_size=1,
                                                        rank=rank)

        suffix = dataset.tag.suffix
        confMat = torch.zeros(self.n_classes, 2, 2).cuda(gpu)

        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader):
                img = batch['image'].cuda(gpu)
                gt = batch['mask'].cuda(gpu)
                probas = model(img)
                probas = torch.sigmoid(probas)
                if self.save_prediction:
                    self.save_predictions(probas, batch)
                model.pr_curve.update(probas.permute((0, 2, 3, 1)).reshape(-1, self.n_classes),
                                      gt.permute((0, 2, 3, 1)).reshape(-1, self.n_classes))
                preds = probas > 0.5
                confMat += NNmetrics.confusion_matrix(preds, gt, num_classes=self.n_classes, multilabel=True)
            if self.multi_gpu:
                confMat = reduce_tensor(confMat, self.world_size, mode='sum')
            precision, recall, thresholds = model.pr_curve.compute()

        if self.is_main_process(rank):
            mIoU = NNmetrics.mIoU_cm(confMat)
            stats = NNmetrics.report_cm(confMat)
            stats['mIoU'] = mIoU
            test_scores = {}
            for l, p, r in zip(LESIONS_LABELS, precision, recall):
                test_scores[l + '_roc' + suffix] = self.auc(r, p).item()
            for k in stats:
                test_scores['Test_%s' % k + suffix] = stats[k]
            self.log_metrics(step=0, **test_scores)
            confMat = confMat.cpu().numpy()
            np.save(os.path.join(self.tracker.prediction_savepoint, 'confMat' + suffix + '.npy'), confMat)
            self.log_artifacts(os.path.join(self.tracker.prediction_savepoint, 'confMat' + suffix + '.npy'))

    def inference(self, model, rank, dataset):
        gpu = self.get_gpu_from_rank(rank)
        test_loader, test_sampler = self.get_dataloader(dataset, shuffle=False, batch_size=1,
                                                        rank=rank)
        with torch.no_grad():
            for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
                img = batch['image'].cuda(gpu)
                probas = model(img)
                probas = torch.sigmoid(probas)
                self.save_predictions(probas, batch)

    def save_predictions(self, predicted_probas, batch):
        indices = batch['index']
        filenames = self.test_dataset.filename(indices)

        for ps, f in zip(predicted_probas, filenames):
            for i, p in enumerate(ps):
                folder = os.path.join(self.tracker.prediction_savepoint, LESIONS_LABELS[i])
                if not os.path.exists(folder):
                    os.makedirs(folder)
                out = (p.cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(folder, f), out)

    def validate(self, model, valid_loader, iteration, rank=0, loss_function=None):
        model.eval()
        gpu = self.get_gpu_from_rank(rank)
        confMat = torch.zeros(self.n_classes, 2, 2).cuda(gpu)
        losses = []
        for n, batch in enumerate(valid_loader):
            batch = self.batch_to_device(batch, rank)
            img = batch['image']
            gt = batch['mask']
            proba = model(img)
            losses.append(loss_function(proba, gt).item())
            preds = torch.sigmoid(proba) >= 0.5
            confMat += NNmetrics.confusion_matrix(preds, gt, num_classes=self.n_classes, multilabel=True)

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

    def forward_train(self, model, loss_function, rank, batch):
        if self.contrastive_pretrain:

            iteration = self.ctx_train['iteration']
            if iteration == 1:
                for g in self.ctx['optimizer'].param_groups:
                    g['lr'] = self.c['Contrastive_training']['lr']

            if iteration > self.c['Contrastive_training']['training_step']:
                self.contrastive_pretrain = False
                for g in self.ctx['optimizer'].param_groups:
                    g['lr'] = self.c['Optimizer']['params_solver']['lr']
                # TODO: Find a way to reset the learning rate scheduler

                model.activate_segmentation_mode()
            else:
                self.contrastive_training(model, rank, batch)
        else:
            return super(RetinExp, self).forward_train(model, loss_function, rank, batch)

    def contrastive_training(self, model, rank, batch):
        pass
        # Todo Implement the contrastive training (within image and cross-image)
