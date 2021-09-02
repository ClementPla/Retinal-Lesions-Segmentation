import random

import torch
import torch.nn as nn
from nntools.nnet import register_loss


class MultiLabelSoftBinaryCrossEntropy(nn.Module):
    def __init__(self, smooth_factor: float = 0, weighted: bool = True,
                 mcb: bool = False, hp_lambda: int = 10,
                 epsilon: float = 0.1, logits=True,
                 first_class_bg=False):
        super(MultiLabelSoftBinaryCrossEntropy, self).__init__()
        self.smooth_factor = smooth_factor
        self.logits = logits
        if logits:
            self.criterion = nn.BCEWithLogitsLoss(reduction='none' if weighted else 'mean')
        else:
            self.criterion = nn.BCELoss(reduction='none' if weighted else 'mean')
        self.weighted = weighted
        self.hp_lambda = hp_lambda
        self.MCB = mcb
        self.epsilon = epsilon
        self.first_class_bg = first_class_bg

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.size() != y_true.size():
            """
            Case in which y_pred.shape == b x c+1 x h x w and y_true.shape == b x c x h x w
            """
            y_pred = y_pred[:, 1:]  # We don't consider the first class (assuming it is background)

        b, c, h, w = y_true.shape
        y_true = y_true.float()

        if self.smooth_factor:
            smooth = random.uniform(0, self.smooth_factor)
            soft_targets = (1 - y_true) * smooth + y_true * (1 - smooth)
        else:
            soft_targets = y_true

        bce_loss = self.criterion(y_pred, soft_targets)

        if self.weighted and not self.MCB:
            N = h * w
            weights = y_true.sum(dim=(2, 3), keepdim=True) / N
            betas = 1 - weights
            bce_loss = y_true * bce_loss * betas + (1 - y_true) * bce_loss * weights
            bce_loss = bce_loss.sum() / (b * N)

        if self.weighted and self.MCB:
            Ypos = y_true.sum(dim=(0, 2, 3), keepdim=False)
            mcb_loss = 0
            for i, k in enumerate(Ypos):
                if self.first_class_bg and i == 0:
                    tmp = (y_true[:, i] * bce_loss[:, i]).flatten(1, 2)
                    mcb_loss += torch.topk(tmp, k=self.hp_lambda*25, dim=1, sorted=False).values.mean()

                else:
                    tmp = ((1 - y_true[:, i]) * bce_loss[:, i]).flatten(1, 2)
                    topk = max(min((k * self.hp_lambda) // b, (1 - y_true[:, i]).sum() // b), self.hp_lambda)
                    ik = torch.topk(tmp, k=int(topk), dim=1, sorted=False).values
                    # We can't compute a "k" per image on the batch, so we take an average value
                    # (limitation of the topk function)

                    beta_k = (ik.shape[1] / (k/b + ik.shape[1] + self.epsilon))
                    # For the same reason, beta_k is batch-wise, not image-wise.
                    # The original paper defines a single beta instead of beta_k; the rational of this choice is unclear.
                    # On the other hand, here beta_k=lambda/(1+lambda)
                    mcb_loss += (ik * (1 - beta_k)).mean()  # Negative loss
                    tmp = y_true[:, i] * bce_loss[:, i]  # Positive Loss
                    mcb_loss += (tmp * beta_k).sum() / (y_true[:, i].sum() + self.epsilon)
            bce_loss = mcb_loss

        return bce_loss


register_loss('MultiLabelSoftBinaryCrossEntropy', MultiLabelSoftBinaryCrossEntropy)


class MultiDatasetCrossEntropy(nn.Module):
    def __init__(self, smooth_factor: float = 0, weighted: bool = True, mcb: bool = False,
                 hp_lambda: int = 10, alpha=None,
                 epsilon: float = 1e-5,
                 criterion='CustomCrossEntropy'):
        super(MultiDatasetCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.criterion = criterion
        if self.criterion == 'CustomCrossEntropy':
            self.loss = MultiLabelSoftBinaryCrossEntropy(smooth_factor, weighted, mcb, hp_lambda, epsilon, logits=False,
                                                         first_class_bg=True)
        elif self.criterion == 'NLL':
            self.loss = nn.NLLLoss()

    def forward(self, y_pred: torch.Tensor, cm_predictions: list, tag: torch.Tensor, y_true: torch.Tensor):
        """
        :param y_pred: Estimation of real labels, BxCxHxW
        :param cm_predictions: List of predicted confusion matrix. Each element of the list is a tuple.
        tuple[0] = dataset id, tuple[1] = CM tensor of size BxC**2xHxW
        :param tag: Tensor of size B
        :param y_true: Labels associated to each image (depends of the dataset): BxCxHxW
        :return:
        """
        loss = 0.0
        regularization = 0.0
        # y_pred = torch.softmax(y_pred, 1)
        y_pred = torch.sigmoid(y_pred)
        y_background = torch.clamp(1-y_pred.max(1, keepdim=True).values, 0, 1)
        y_pred = torch.cat([y_background, y_pred], 1)

        y_bg_true = ~torch.any(y_true, 1, keepdim=True).long()
        y_true = torch.cat([y_bg_true, y_true], 1)
        
        if self.criterion == 'NLL':
            max_arg = torch.max(y_true, 1, keepdim=False)
            gt = max_arg.indices + 1
            gt[max_arg.values == 0] = 0
            y_true = gt

        for d_id, cm in cm_predictions:
            y_true_did = y_true[tag == d_id]
            y_pred_did = y_pred[tag == d_id]
            b, c, h, w = y_pred_did.shape
            y_pred_did = y_pred_did.view(b, c, h*w).permute(0, 2, 1).reshape(b*h*w, c, 1)
            cm = cm.view(b, c**2, h*w).permute(0, 2, 1)
            cm = cm.reshape(b*h*w, c**2).view(b*h*w, c, c)
            cm = cm / (cm.sum(1, keepdim=True)+self.epsilon)
            # cm = torch.sigmoid(cm)
            y_pred_n = torch.bmm(cm, y_pred_did).view(b*h*w, c)
            y_pred_n = y_pred_n.view(b, h*w, c).permute(0, 2, 1).reshape(b, c, h, w)
            loss += self.loss(torch.clamp(y_pred_n, self.epsilon, 1), y_true_did)
            regularization += torch.trace(torch.sum(cm, dim=0)) / (b*h*w)
        return loss + self.alpha*regularization


register_loss('MultiDatasetCrossEntropy', MultiDatasetCrossEntropy)


class MultiLabelToCrossEntropyLoss(nn.Module):
    def __init__(self, from_logits=False):
        super(MultiLabelToCrossEntropyLoss, self).__init__()

        if from_logits:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

    def forward(self, y_pred, y_true):
        max_arg = torch.max(y_true, 1, keepdim=False)
        gt = max_arg.indices + 1
        gt[max_arg.values == 0] = 0
        y_true = gt
        return self.loss(y_pred, y_true)
