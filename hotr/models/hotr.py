# ------------------------------------------------------------------------
# HOTR official code : hotr/models/hotr.py
# Copyright (c) Kakao Brain, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
import datetime
import tensorflow as tf


from hotr.util.misc import NestedTensor, nested_tensor_from_tensor_list
from .feed_forward import MLP
from hotr.util.box_ops import BoxRelationalEmbedding
import os
import numpy as np

class HOTR(nn.Module):
    def __init__(self, detr,
                 num_hoi_queries,
                 num_actions,
                 interaction_transformer,
                 freeze_detr,
                 share_enc,
                 pretrained_dec,
                 temperature,
                 hoi_aux_loss,
                 return_obj_class=None):
        super().__init__()

        # * Instance Transformer ---------------
        self.dim_fasttext = 300
        self.dim_clip = 512
        self.linear = nn.Linear(self.dim_fasttext, 256)
        self.linear2 = nn.Linear(self.dim_clip, 256)
        self.detr = detr
        if freeze_detr:
            # if this flag is given, freeze the object detection related parameters of DETR
            for p in self.parameters():
                p.requires_grad_(False)
        hidden_dim = detr.transformer.d_model
        # --------------------------------------

        # * Interaction Transformer -----------------------------------------
        self.num_queries = num_hoi_queries
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        self.H_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.O_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.action_embed = nn.Linear(hidden_dim, num_actions+1)
        # --------------------------------------------------------------------

        # * HICO-DET FFN heads ---------------------------------------------
        self.return_obj_class = (return_obj_class is not None)
        if return_obj_class: self._valid_obj_ids = return_obj_class + [return_obj_class[-1]+1]
        # ------------------------------------------------------------------

        # * Transformer Options ---------------------------------------------
        self.interaction_transformer = interaction_transformer

        if share_enc: # share encoder
            self.interaction_transformer.encoder = detr.transformer.encoder

        if pretrained_dec: # free variables for interaction decoder
            self.interaction_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in self.interaction_transformer.decoder.parameters():
                p.requires_grad_(True)
        # ---------------------------------------------------------------------

        # * Loss Options -------------------
        self.tau = temperature
        self.hoi_aux_loss = hoi_aux_loss
        # ----------------------------------
        self.semantic_features_path = '/vcoco/v-coco/semanticFeatures'
        self.caption_feature_path = '/vcoco/v-coco/captions_npy'

    def forward(self, samples: NestedTensor, image_ids: list):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        # >>>>>>>>>>>>  BACKBONE LAYERS  <<<<<<<<<<<<<<<
        # bs, _, _ = features.shape   
        features, pos = self.detr.backbone(samples) # H*W, bs, dim / bs, H*W, dim
        c5, mask = features[-1].decompose() # src: feature input fitting into DETR

        assert mask is not None
        # ----------------------------------------------

        # >>>>>>>>>>>>  LOAD SEMANTIC FEATURES  <<<<<<<<<<<<<<<
        num_pad = c5.shape[-1] * c5.shape[-2]
        semantic_features = []
        semantic_masks = []
        caption_features = []
        caption_masks = []
        for image_id in image_ids:
            try:
              image_id_str = image_id.cpu().numpy()
              image_id_str = tf.compat.as_str_any(image_id_str[0])
              filename = 'COCO_train2014_' + str(str(image_id_str).zfill(12)) + '.npy'              

              semantic_feature = np.load(os.path.join(self.semantic_features_path, filename))
              caption_feature = np.load(os.path.join(self.caption_feature_path, filename))
            except:
              image_id_str = image_id.cpu().numpy()
              image_id_str = tf.compat.as_str_any(image_id_str[0])
              filename = 'COCO_val2014_' + str(str(image_id_str).zfill(12)) + '.npy'

              semantic_feature = np.load(os.path.join(self.semantic_features_path, filename))
              caption_feature = np.load(os.path.join(self.caption_feature_path, filename))
            num_objects, _ = semantic_feature.shape
            semantic_feature = np.concatenate((semantic_feature, np.zeros((num_pad - num_objects, self.dim_fasttext))), axis=0)
            semantic_features.append(semantic_feature)
            semantic_mask = [False] * num_objects + [True] * (num_pad - num_objects)
            semantic_masks.append(semantic_mask)

            caption_feature = np.concatenate((caption_feature, np.zeros((num_pad - 1, self.dim_clip))), axis=0)
            caption_features.append(caption_feature)
            caption_mask = [False] * 1 + [True] * (num_pad - 1)
            caption_masks.append(caption_mask)
        
        semantic_features = torch.FloatTensor(semantic_features).to('cuda:0')
        semantic_features = self.linear(semantic_features)
        semantic_features = semantic_features.permute(1, 0, 2)
        semantic_masks = torch.Tensor(semantic_masks).to('cuda:0')

        caption_features = torch.FloatTensor(caption_features).to('cuda:0')
        caption_features = self.linear2(caption_features)
        caption_features = caption_features.permute(1, 0, 2)
        caption_masks = torch.Tensor(caption_masks).to('cuda:0')
        # >>>>>>>>>>>> OBJECT DETECTION LAYERS <<<<<<<<<<
        start_time = time.time()
        # Fitting into DETR for object detection
        hs, memory = self.detr.transformer(self.detr.input_proj(c5), mask, self.detr.query_embed.weight, pos[-1], semantic_features=None, semantic_masks=None, caption_features=None, caption_masks=None)
        inst_repr = F.normalize(hs[-1], p=2, dim=2) # instance representations
    
        # Prediction Heads for Object Detection
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        object_detection_time = time.time() - start_time

        # -----------------------------------------------

        # >>>>>>>>>>>> HOI DETECTION LAYERS <<<<<<<<<<<<<<<
        start_time = time.time()
        assert hasattr(self, 'interaction_transformer'), "Missing Interaction Transformer."
        interaction_hs = self.interaction_transformer(self.detr.input_proj(c5), mask, self.query_embed.weight, pos[-1], semantic_features=semantic_features, semantic_masks=semantic_masks, caption_features = caption_features, caption_masks=caption_masks)[0] # interaction representations
        # [HO Pointers]
        H_Pointer_reprs = F.normalize(self.H_Pointer_embed(interaction_hs), p=2, dim=-1)
        O_Pointer_reprs = F.normalize(self.O_Pointer_embed(interaction_hs), p=2, dim=-1)

        # Take average of encoder outputs
        n = 3 
        outputs_hidx = []
        for H_Pointer_repr in H_Pointer_reprs:
          inst_repr = F.normalize(hs[-1], p=2, dim=2)
          output_hidx = (torch.bmm(H_Pointer_repr, inst_repr.transpose(1, 2))) / self.tau
          for i in range(2, n + 1):
            inst_repr = F.normalize(hs[-i], p=2, dim=2)
            output_hidx += ((torch.bmm(H_Pointer_repr, inst_repr.transpose(1, 2))) / self.tau)
          output_hidx /= n
          outputs_hidx.append(output_hidx)

        outputs_oidx = []
        for O_Pointer_repr in O_Pointer_reprs:
          inst_repr = F.normalize(hs[-1], p=2, dim=2)
          output_oidx = (torch.bmm(O_Pointer_repr, inst_repr.transpose(1, 2))) / self.tau
          for i in range(2, n + 1):
            inst_repr = F.normalize(hs[-i], p=2, dim=2)
            output_oidx += ((torch.bmm(O_Pointer_repr, inst_repr.transpose(1, 2))) / self.tau)
          output_oidx /= n
          outputs_oidx.append(output_oidx)
        
        # [Action Classification]
        outputs_action = self.action_embed(interaction_hs)
        # --------------------------------------------------
        hoi_detection_time = time.time() - start_time
        hoi_recognition_time = max(hoi_detection_time - object_detection_time, 0)
        # -------------------------------------------------------------------

        # [Target Classification]
        if self.return_obj_class:
            detr_logits = outputs_class[-1, ..., self._valid_obj_ids]
            o_indices = [output_oidx.max(-1)[-1] for output_oidx in outputs_oidx]
            obj_logit_stack = [torch.stack([detr_logits[batch_, o_idx, :] for batch_, o_idx in enumerate(o_indice)], 0) for o_indice in o_indices]
            outputs_obj_class = obj_logit_stack

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_hidx": outputs_hidx[-1],
            "pred_oidx": outputs_oidx[-1],
            "pred_actions": outputs_action[-1],
            "hoi_recognition_time": hoi_recognition_time,
        }

        if self.return_obj_class: out["pred_obj_logits"] = outputs_obj_class[-1]

        if self.hoi_aux_loss: # auxiliary loss
            out['hoi_aux_outputs'] = \
                self._set_aux_loss_with_tgt(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_obj_class) \
                if self.return_obj_class else \
                self._set_aux_loss(outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e}
                for a, b, c, d, e in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1])]

    @torch.jit.unused
    def _set_aux_loss_with_tgt(self, outputs_class, outputs_coord, outputs_hidx, outputs_oidx, outputs_action, outputs_tgt):
        return [{'pred_logits': a,  'pred_boxes': b, 'pred_hidx': c, 'pred_oidx': d, 'pred_actions': e, 'pred_obj_logits': f}
                for a, b, c, d, e, f in zip(
                    outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                    outputs_hidx[:-1],
                    outputs_oidx[:-1],
                    outputs_action[:-1],
                    outputs_tgt[:-1])]
