# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Part of the code is from https://github.com/tztztztztz/eql.detectron2/blob/master/projects/EQL/eql/fast_rcnn.py
import logging
import math
import json
from typing import Dict, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.utils.comm import get_world_size

__all__ = ["CustomFastRCNNOutputLayers", "CustomFastRCNNOutputs"]
class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # "gt_classes" exists if and only if training. But other gt fields may
            # not necessarily exist in training for images that have no groundtruth.
            if proposals[0].has("gt_classes"):
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = [
                    p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes for p in proposals
                ]
                self.gt_boxes = box_type.cat(gt_boxes)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  # no instances found

    def softmax_cross_entropy_loss(self):
        """
        Deprecated
        """
        _log_classification_stats(self.pred_class_logits, self.gt_classes)
        return cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):
        """
        Deprecated
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.proposals.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds should produce a valid loss of zero because reduction=sum.
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * self.gt_classes[fg_inds, None] + torch.arange(
                box_dim, device=device
            )

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                self.proposals.tensor[fg_inds],
            )
            loss_box_reg = giou_loss(
                fg_pred_boxes,
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Deprecated
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

    def predict_boxes(self):
        """
        Deprecated
        """
        pred = self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)
        return pred.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)


class CustomFastRCNNOutputs(FastRCNNOutputs):
    def __init__(
        self,
        cfg,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        freq_weight=None,
    ):
        super().__init__(box2box_transform, pred_class_logits, 
            pred_proposal_deltas, proposals, smooth_l1_beta, box_reg_loss_type)
        self._no_instances = (self.pred_class_logits.numel() == 0) or (len(proposals) == 0)
        if self._no_instances:
            print('No instances!', pred_class_logits.shape, pred_proposal_deltas.shape, len(proposals))
        self.box_batch_size = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE * len(proposals)
        self.use_sigmoid_ce = cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE
        self.use_eql_loss = cfg.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS
        self.use_fed_loss = cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS
        self.fed_loss_num_cat = cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT
        self.fed_loss_freq_weight = cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT
        self.freq_weight = freq_weight

        if len(self.gt_classes) > 0: 
            assert self.gt_classes.max() <= cfg.MODEL.ROI_HEADS.NUM_CLASSES, self.gt_classes.max()

    def sigmoid_cross_entropy_loss(self):
        if self._no_instances:
            return self.pred_class_logits.new_zeros([1])[0] # This is more robust than .sum() * 0.
        # self._log_accuracy()
        _log_classification_stats(self.pred_class_logits, self.gt_classes)

        B = self.pred_class_logits.shape[0]
        C = self.pred_class_logits.shape[1] - 1

        target = self.pred_class_logits.new_zeros(B, C + 1)
        target[range(len(self.gt_classes)), self.gt_classes] = 1 # B x (C + 1)
        target = target[:, :C] # B x C

        weight = 1
        if (self.freq_weight is not None) and self.use_eql_loss: # eql loss
            exclude_weight = (self.gt_classes != C).float().view(B, 1).expand(B, C)
            threshold_weight = self.freq_weight.view(1, C).expand(B, C)
            eql_w = 1 - exclude_weight * threshold_weight * (1 - target) # B x C
            weight = weight * eql_w

        if (self.freq_weight is not None) and self.use_fed_loss: # fedloss
            appeared = torch.unique(self.gt_classes) # C'
            prob = appeared.new_ones(C + 1).float()
            if len(appeared) < self.fed_loss_num_cat:
                if self.fed_loss_freq_weight > 0:
                    prob[:C] = self.freq_weight.float().clone()
                else:
                    prob[:C] = prob[:C] * (1 - self.freq_weight)
                prob[appeared] = 0
                more_appeared = torch.multinomial(
                    prob, self.fed_loss_num_cat - len(appeared),
                    replacement=False)
                appeared = torch.cat([appeared, more_appeared])
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1 # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w

        cls_loss = F.binary_cross_entropy_with_logits(
            self.pred_class_logits[:, :-1], target, reduction='none') # B x C
        return torch.sum(cls_loss * weight) / B


    def softmax_cross_entropy_loss(self):
        """
        change _no_instance handling
        """
        if self._no_instances:
            return self.pred_class_logits.new_zeros([1])[0]
        else:
            # self._log_accuracy()
            _log_classification_stats(self.pred_class_logits, self.gt_classes)
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")


    def box_reg_loss(self):
        """
        change _no_instance handling and normalization
        """
        if self._no_instances:
            print('No instance in box reg loss')
            return self.pred_proposal_deltas.new_zeros([1])[0]

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss()
        else:
            loss_cls = self.softmax_cross_entropy_loss()
        return {
            "loss_cls": loss_cls, 
            "loss_box_reg": self.box_reg_loss()
        }
        
    def predict_probs(self):
        """
        Deprecated
        """
        if self.use_sigmoid_ce:
            probs = F.sigmoid(self.pred_class_logits)
        else:
            probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)


def _load_class_freq(cfg):
    freq_weight = None
    if cfg.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS or cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS:
        cat_info = json.load(open(cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH, 'r'))
        cat_info = torch.tensor(
            [c['image_count'] for c in sorted(cat_info, key=lambda x: x['id'])],
            device=torch.device(cfg.MODEL.DEVICE))
        if cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS and \
            cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT > 0.:
            freq_weight = \
                cat_info.float() ** cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT
        else:
            thresh, _ = torch.kthvalue(
                cat_info,
                len(cat_info) - cfg.MODEL.ROI_BOX_HEAD.EQL_FREQ_CAT + 1)
            freq_weight = (cat_info < thresh.item()).float()

    return freq_weight


class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(
        self, 
        cfg, 
        input_shape: ShapeSpec,
        **kwargs
    ):
        super().__init__(cfg, input_shape, **kwargs)
        self.use_sigmoid_ce = cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE
        if self.use_sigmoid_ce:
            prior_prob = cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_score.bias, bias_value)
        
        self.cfg = cfg
        self.freq_weight = _load_class_freq(cfg)

    def losses(self, predictions, proposals, use_advanced_loss=True):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions
        losses = CustomFastRCNNOutputs(
            self.cfg,
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            self.freq_weight if use_advanced_loss else None, 
        ).losses()
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions, proposals):
        """
        enable use proposal boxes
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        if self.cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE:
            proposal_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * ps[:, None]) ** 0.5 \
                for s, ps in zip(scores, proposal_scores)]
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )


    def predict_probs(self, predictions, proposals):
        """
        support sigmoid
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
