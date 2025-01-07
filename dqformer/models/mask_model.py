import dqformer.utils.testing as testing
import MinkowskiEngine as ME
import torch
import torch.nn.functional as F
import numpy as np
import time
from dqformer.models.decoder import MaskedTransformerDecoder
from dqformer.models.loss import MaskLoss_st, ThingLoss, SemLoss
from dqformer.utils.evaluate_panoptic import PanopticEvaluator
from pytorch_lightning.core.lightning import LightningModule
from dqformer.models.backbone import SphereFormer_Backbone
from dqformer.models.thing_neck import RPN_multitask
from dqformer.models.stuff_neck import RPN_with_Decoder
from dqformer.utils.visulize import vis_instance_seg, vis_semantic_seg, vis_heatmap, vis_points_with_axis

class MaskPS(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(dict(hparams))
        self.cfg = hparams

        self.backbone = SphereFormer_Backbone()
        state_dict = torch.load('../ckpts/dqformer_backbone_kitti.pth', map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=False)

        self.thing_neck = RPN_multitask(dataset=self.cfg[self.cfg.MODEL.DATASET])
        self.stuff_neck = RPN_with_Decoder(hparams.BACKBONE, hparams.DECODER)

        self.decoder = MaskedTransformerDecoder(
            hparams.DECODER,
            hparams.BACKBONE,
            hparams[hparams.MODEL.DATASET],
        )

        self.stuff_loss = MaskLoss_st(hparams.LOSS, hparams[hparams.MODEL.DATASET], hparams.DECODER.USE_MATCHER, self.backbone.stuff_class)
        self.thing_loss = ThingLoss(hparams.LOSS)
        self.sem_loss = SemLoss(hparams.LOSS.SEM.WEIGHTS)

        self.evaluator = PanopticEvaluator(
            hparams[hparams.MODEL.DATASET], hparams.MODEL.DATASET
        )

    def forward(self, x):
        vox_feat, feats, coors, stuff_queries, pad_masks, bb_logits = self.backbone(x)
        # Thing Query Generator
        Thing_dict_list = self.thing_neck(vox_feat, example=x['Th_dict'] if 'Th_dict' in x.keys() else None)
        # Stuff Query Generator
        stuff_queries, stuff_loss_dict = self.stuff_neck(vox_feat, stuff_queries, example=x['bev_sem'] if 'bev_sem' in x.keys() else None)
        # Query-Orinted Mask Decoder
        outputs_stuff, outputs_thing, padding = self.decoder(feats, coors, pad_masks, stuff_queries, Thing_dict_list)
        return outputs_stuff, outputs_thing, bb_logits, padding, stuff_loss_dict

    def getLoss(self, x, outputs_stuff, outputs_thing, bb_logits, padding):
        # stuff loss
        targets = {"classes": x["masks_cls_st"], "masks": x["masks_st"]}
        loss_mask = self.stuff_loss(outputs_stuff["out_st"], targets, x["masks_ids_st"], x["pt_coord"])
        # thing loss
        Th_targets = x["Th_dict"]
        loss_thing = self.thing_loss(outputs_thing, Th_targets, x["ins_masks_ids"])
        loss_mask.update(loss_thing)

        sem_labels = [
            torch.from_numpy(i).type(torch.LongTensor).cuda() for i in x["sem_label"]
        ]
        sem_labels = torch.cat([s.squeeze(1) for s in sem_labels], dim=0)
        bb_logits = bb_logits[~padding]
        loss_sem_bb = self.sem_loss(bb_logits, sem_labels)
        loss_mask.update(loss_sem_bb)

        return loss_mask

    def training_step(self, x: dict, idx):
        outputs_stuff, outputs_thing, bb_logits, padding, stuff_loss_dict = self.forward(x)
        loss_dict = self.getLoss(x, outputs_stuff, outputs_thing, bb_logits, padding)
        loss_dict = {key: value for d in [stuff_loss_dict, loss_dict] for key, value in d.items()}  # merge stuff_loss_dict and loss_dcit
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        total_loss = sum(loss_dict.values())
        self.log("train_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        # torch.cuda.empty_cache()

        return total_loss

    def validation_step(self, x: dict, idx):
        if "EVALUATE" in self.cfg:
            self.evaluation_step(x, idx)
            return
        outputs_stuff, outputs_thing, bb_logits, padding, stuff_loss_dict = self.forward(x)
        loss_dict = self.getLoss(x, outputs_stuff, outputs_thing, bb_logits, padding)
        loss_dict = {key: value for d in [stuff_loss_dict, loss_dict] for key, value in d.items()}  # merge stuff_loss_dict and loss_dcit
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, batch_size=self.cfg.TRAIN.BATCH_SIZE)
        total_loss = sum(loss_dict.values())
        self.log("val_loss", total_loss, batch_size=self.cfg.TRAIN.BATCH_SIZE)

        sem_pred, ins_pred = self.panoptic_inference(outputs_stuff, outputs_thing, padding)
        self.evaluator.update(sem_pred, ins_pred, x)

        # torch.cuda.empty_cache()
        return total_loss

    def validation_epoch_end(self, outputs):
        bs = self.cfg.TRAIN.BATCH_SIZE
        self.log("metrics/pq", self.evaluator.get_mean_pq(), batch_size=bs)
        self.log("metrics/iou", self.evaluator.get_mean_iou(), batch_size=bs)
        self.log("metrics/rq", self.evaluator.get_mean_rq(), batch_size=bs)
        if not "EVALUATE" in self.cfg:
            self.evaluator.reset()

    def evaluation_step(self, x: dict, idx):
        outputs_stuff, outputs_thing, bb_logits, padding, stuff_loss_dict = self.forward(x)
        sem_pred, ins_pred = self.panoptic_inference(outputs_stuff, outputs_thing, padding)
        self.evaluator.update(sem_pred, ins_pred, x)

    def test_step(self, x: dict, idx):
        outputs_stuff, outputs_thing, bb_logits, padding, stuff_loss_dict = self.forward(x)
        sem_pred, ins_pred = self.panoptic_inference(outputs_stuff, outputs_thing, padding)

        if "RESULTS_DIR" in self.cfg:
            results_dir = self.cfg.RESULTS_DIR
            class_inv_lut = self.evaluator.get_class_inv_lut()
            dt = self.cfg.MODEL.DATASET
            testing.save_results(
                sem_pred, ins_pred, results_dir, x[0] if self.cfg[self.cfg.MODEL.DATASET]['USE_TTA'] else x,
                    class_inv_lut, x[0]["token"] if self.cfg[self.cfg.MODEL.DATASET]['USE_TTA'] else x["token"], dt
            )
        # torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.cfg.TRAIN.STEP, gamma=self.cfg.TRAIN.DECAY
        )
        return [optimizer], [scheduler]

    def semantic_inference(self, outputs, padding):
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        semseg = []
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred[~pad].sigmoid()  # throw padding points
            pred = torch.einsum("qc,pq->pc", mask_cls, mask_pred)
            semseg.append(torch.argmax(pred, dim=1))
        return semseg

    def panoptic_inference(self, outputs_stuff, outputs_things, padding):
        bs = outputs_stuff["out_st"]["pred_logits"].shape[0]
        th_masks, th_masks_bin, th_labels, th_scores = [], [], [], []
        for b_id in range(bs):
            batch_mask = torch.cat([th_pred['Thing_pred_masks'][b_id][~padding[b_id]].sigmoid().transpose(0,1) for th_pred in outputs_things], dim=0) # Nq, Np
            # merge Thing cls by tasks
            flag = 1
            batch_label = []
            for task_id, th_pred in enumerate(outputs_things):
                cur_labels = th_pred['labels'][b_id] + flag  # Nq
                batch_label.append(cur_labels)
                flag += self.decoder.Thing_tasks[task_id]['num_class']
            batch_label = torch.cat(batch_label, dim=0) # Nq
            batch_score = torch.cat([th_pred['scores'][b_id] for th_pred in outputs_things], dim=0)    # Nq

            batch_select = torch.cat([th_pred['mask'][b_id] for th_pred in outputs_things], dim=0)    # Nq
            batch_mask = batch_mask[batch_select]
            batch_label = batch_label[batch_select]
            batch_score = batch_score[batch_select]

            if batch_mask.shape[0] == 0:    # no object
                th_masks.append(batch_mask)
                th_masks_bin.append(batch_mask)
                th_labels.append(batch_label)
                th_scores.append(batch_score)
            else:
                selected_mask, selected_bin_mask, selected_label, selected_scores = self.post_processing(batch_mask, batch_label)
                keep = (selected_bin_mask.sum(1) > 0) & (selected_scores > 0.8)
                th_masks.append(selected_mask[keep])
                th_masks_bin.append(selected_bin_mask[keep])
                th_labels.append(selected_label[keep])
                th_scores.append(selected_scores[keep])

        mask_cls = outputs_stuff["out_st"]["pred_logits"]
        mask_pred = outputs_stuff["out_st"]["pred_masks"]
        things_ids = self.trainer.datamodule.things_ids
        num_classes = self.cfg[self.cfg.MODEL.DATASET].NUM_CLASSES
        sem_pred = []
        ins_pred = []
        panoptic_output = []
        info = []
        for mask_cls, mask_pred, th_mask, th_mask_bin, th_label, pad in zip(mask_cls, mask_pred, th_masks, th_masks_bin, th_labels, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()
            keep = labels.ne(num_classes)

            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            cur_mask_cls = mask_cls[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            # prob to belong to each of the `keep` masks for each point
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks

            panoptic_seg = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            th_masks = []
            th_masks_bin = []
            st_masks = []
            th_segments_info = []
            st_segments_info = []
            segment_id = 0
            if cur_masks.shape[1] == 0:  # no masks detected
                panoptic_output.append(panoptic_seg)
                info.append([])
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                # mask index for each point: between 0 and (`keep` - 1)
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()  # current class
                    isthing = pred_class in things_ids
                    mask_area = (cur_mask_ids == k).sum().item()  # points in mask k
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()  # binary mas
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)

                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.cfg.MODEL.OVERLAP_THRESHOLD:
                            continue  # binary mask occluded 80%
                        if not isthing:  # merge stuff regions
                            if int(pred_class) in stuff_memory_list.keys():
                                # in the list, asign id stored on the list for that class
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                # not in the list, class = cur_id + 1
                                stuff_memory_list[int(pred_class)] = segment_id + 1
                        segment_id += 1
                        panoptic_seg[mask] = segment_id
                        segment_info = {
                                "id": segment_id,
                                "isthing": bool(isthing),
                                "sem_class": pred_class,
                        }
                        if isthing:
                            th_masks.append(cur_masks[:, k][None])
                            th_masks_bin.append(mask[None])
                            th_segments_info.append(segment_info)
                        else:
                            st_masks.append(mask)
                            st_segments_info.append(segment_info)
                panoptic_output.append(panoptic_seg)

                # concat  dynamic-th & kernel-th
                if len(th_masks) != 0:  # dynamic + Thing
                    th_masks = torch.cat(th_masks, dim=0)
                    # th_masks = torch.cat([th_masks, th_mask], dim=0)    # concat dynamic & Thing
                    th_masks_bin = torch.cat(th_masks_bin, dim=0).float()
                    # th_masks_bin = torch.cat([th_masks_bin, th_mask_bin], dim=0)    # concat dynamic & Thing
                    th_labels = torch.tensor([info['sem_class'] for info in th_segments_info]).to(th_label)
                    # th_labels = torch.cat([th_labels, th_label])    # concat dynamic & Thing
                    # dynamic & Thing -> NMS
                    # th_masks_after, th_masks_bin_after, th_labels_after, _ = self.post_processing(th_masks, th_labels, th_masks_bin)
                elif th_mask.shape[0] != 0:
                    th_masks = th_mask
                    th_masks_bin = th_mask_bin
                    th_labels = th_label
                else:
                    th_masks_bin = []
                    th_labels = []


                # things
                for id, (mask, label) in enumerate(zip(th_masks_bin, th_labels)):
                    sem[mask.bool()] = int(label)
                    ins[mask.bool()] = id + 1
                # stuff
                for mask, inf in zip(st_masks, st_segments_info):
                    sem[mask] = inf["sem_class"]
                    ins[mask] = 0
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())

        return sem_pred, ins_pred

    def post_processing(self, proposals_pred, classes, proposals_pred_bin=None, cls_scores=None, threshold=0.85):
        def nms(proposals_pred, proposals_pred_bin, cls_scores, threshold=0.75):
            if proposals_pred_bin is None:
                proposals_pred_bin = (proposals_pred >= 0.5).float()  # (nProposal, N), float, cuda
            proposals_scores = torch.sum(proposals_pred.mul(proposals_pred_bin), dim=1) / (torch.sum(proposals_pred_bin, dim=1) + 1e-6)
            if cls_scores is not None:
                proposals_scores *= cls_scores  # nProposal
            proposals_scores, indices = torch.sort(proposals_scores, descending=True)
            proposals_scores = proposals_scores[indices]
            proposals_pred_bin = proposals_pred_bin[indices]

            intersection = proposals_pred_bin @ proposals_pred_bin.T  # (nProposal, nProposal), float, cuda
            proposals_pointnum = proposals_pred_bin.sum(1)  # (nProposal), float, cuda
            cross_ious = intersection / (proposals_pointnum[None] + proposals_pointnum[:, None] - intersection + 1e-6)

            proposals_pred_bin = proposals_pred_bin.cpu().numpy()
            cross_ious = cross_ious.cpu().numpy()
            ixs = np.arange(proposals_pred_bin.shape[0])
            pick = []
            while len(ixs) > 0:
                i = ixs[0]
                pick.append(i)
                iou = cross_ious[i, ixs[1:]]
                remove_ixs = np.where(iou > threshold)[0] + 1
                ixs = np.delete(ixs, remove_ixs)
                ixs = np.delete(ixs, 0)

            selected_proposals = proposals_pred[pick]
            selected_proposals_bin = torch.from_numpy(proposals_pred_bin[pick]).to(selected_proposals)
            selected_classes = classes[pick]
            selected_scores = proposals_scores[pick]
            return selected_proposals, selected_proposals_bin, selected_classes, selected_scores

        use_nms = False
        if use_nms:
            return nms(proposals_pred, proposals_pred_bin, cls_scores, threshold)

        def IA(proposals, proposals_bin, classes, scores, label_matrix):
            cum_matrix = torch.cumsum(label_matrix.float(), dim=0) < 2
            keep_matrix = cum_matrix.diagonal(0)
            label_matrix = (label_matrix[keep_matrix] & cum_matrix[keep_matrix]).float()

            selected_proposals = torch.mm(label_matrix, proposals.float()) / (torch.mm(label_matrix, proposals_bin.float()) + 1e-6)
            selected_proposals = torch.clamp_max(selected_proposals, 1.0)
            selected_proposals_bin = (torch.mm(label_matrix, proposals_bin.float()) > 0).float()  # selected,Np
            selected_classes = classes[keep_matrix]
            selected_scores = scores[keep_matrix]
            return selected_proposals, selected_proposals_bin, selected_classes, selected_scores

        proposals_list = []
        proposals_bin_list = []
        classes_list = []
        scores_list = []
        for c_id in classes.unique():
            c_valid = classes == c_id

            c_proposals = proposals_pred[c_valid]
            if proposals_pred_bin is not None:
                c_proposals_bin = proposals_pred_bin[c_valid]
            else:
                c_proposals_bin = (c_proposals >= 0.5).float()
            c_proposals_scores = torch.sum(c_proposals.mul(c_proposals_bin), dim=1) / (torch.sum(c_proposals_bin, dim=1) + 1e-6)
            if cls_scores is not None:
                c_proposals_scores *= cls_scores[c_valid]  # nProposal

            intersection = c_proposals_bin @ c_proposals_bin.T  # (nProposal, nProposal), float, cuda
            proposals_pointnum = c_proposals_bin.sum(1)  # (nProposal), float, cuda
            cross_ious = intersection / (proposals_pointnum[None] + proposals_pointnum[:, None] - intersection + 1e-6)
            label_matrix = (cross_ious > threshold)
            merged_c_proposals, merged_c_proposals_bin, merged_c_class, merged_c_scores = \
                IA(c_proposals, c_proposals_bin, classes[c_valid], c_proposals_scores, label_matrix)

            proposals_list.append(merged_c_proposals)
            proposals_bin_list.append(merged_c_proposals_bin)
            classes_list.append(merged_c_class)
            scores_list.append(merged_c_scores)

        return torch.cat(proposals_list, dim=0), torch.cat(proposals_bin_list, dim=0), torch.cat(classes_list, dim=0), torch.cat(scores_list, dim=0)
