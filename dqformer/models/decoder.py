import dqformer.models.blocks as blocks
import torch
from dqformer.models.positional_encoder import PositionalEncoder
from dqformer.utils.visulize import vis_centers
from torch import nn


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, cfg, bb_cfg, data_cfg):
        super().__init__()
        hidden_dim = cfg.HIDDEN_DIM

        cfg.POS_ENC.FEAT_SIZE = cfg.HIDDEN_DIM

        self.pe_layer = PositionalEncoder(cfg.POS_ENC)

        self.num_layers = cfg.FEATURE_LEVELS * cfg.DEC_BLOCKS
        self.nheads = cfg.NHEADS

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                blocks.SelfAttentionLayer(
                    d_model=hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_cross_attention_layers.append(
                blocks.CrossAttentionLayer(
                    d_model=hidden_dim, nhead=self.nheads, dropout=0.0
                )
            )
            self.transformer_ffn_layers.append(
                blocks.FFNLayer(
                    d_model=hidden_dim, dim_feedforward=cfg.DIM_FFN, dropout=0.0
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.Thing_tasks = data_cfg.ASSIGNER.tasks
        self.num_feature_levels = cfg.FEATURE_LEVELS
        
        self.stuff_query_proj = nn.Linear(256, hidden_dim)
        self.query_embed_st = nn.Embedding(11, hidden_dim)

        self.mask_feat_proj = nn.Sequential()
        in_channels = bb_cfg.CHANNELS
        if in_channels[-1] != hidden_dim:
            self.mask_feat_proj = nn.Linear(in_channels[-1], hidden_dim)

        in_channels = in_channels[:-1][-self.num_feature_levels :]
        self.input_proj = nn.ModuleList()
        for ch in in_channels:
            if ch != hidden_dim:  # linear projection to hidden_dim
                self.input_proj.append(nn.Linear(ch, hidden_dim))
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        self.class_embed = nn.Linear(hidden_dim, data_cfg.NUM_CLASSES + 1)
        self.mask_embed = blocks.MLP(hidden_dim, hidden_dim, cfg.HIDDEN_DIM, 3)


    def forward(self, feats, coors, pad_masks, stuff_queries, Thing_dict_list):
        last_coors = coors.pop()
        mask_features = self.mask_feat_proj(feats.pop()) + self.pe_layer(last_coors)
        last_pad = pad_masks.pop()
        src = []
        pos = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(feats[i].shape[1])
            pos.append(self.pe_layer(coors[i]))
            feat = self.input_proj[i](feats[i])
            src.append(feat)

        bs = src[0].shape[0]
        query_embed_st = self.query_embed_st.weight.unsqueeze(0).repeat(bs, 1, 1)
        # query_embed_th = self.query_embed_th.weight.unsqueeze(0).repeat(bs, 1, 1)
        Thing_center_xy = torch.cat([Th["xy"] for Th in Thing_dict_list], dim=1)
        Thing_center_xyz, query_embed_th = self.get_thing_embed(Thing_center_xy, last_coors)
        query_embed = torch.cat([query_embed_st, query_embed_th], dim=1)

        query_st = self.stuff_query_proj(stuff_queries)
        query_th = torch.cat([Th["ct_feat"] for Th in Thing_dict_list], dim=1) # B, tasks*obj_num, C
        output = torch.cat([query_st, query_th], dim=1)

        predictions_class = []
        predictions_mask_st = []
        predictions_mask_th = [] # layers,[taks,[B,Np,obj_num]]

        # predictions on learnable query features, first attn_mask
        outputs_class, attn_mask, st_mask, th_mask = self.pred_heads(
            output,
            mask_features,
            pad_mask=last_pad,
        )
        predictions_class.append(outputs_class)
        predictions_mask_st.append(st_mask)
        predictions_mask_th.append(th_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            if attn_mask is not None:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src[level_index],
                attn_mask=attn_mask,
                padding_mask=pad_masks[level_index],
                pos=pos[level_index],
                query_pos=query_embed,
            )
            output = self.transformer_self_attention_layers[i](
                output, attn_mask=None, padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            # get predictions and attn mask for next feature level
            outputs_class, attn_mask, st_mask, th_mask = self.pred_heads(
                output,
                mask_features,
                pad_mask=last_pad,
            )
            predictions_class.append(outputs_class)
            predictions_mask_st.append(st_mask)
            predictions_mask_th.append(th_mask)

        assert len(predictions_class) == self.num_layers + 1

        out_st = {"pred_logits": predictions_class[-1], "pred_masks": predictions_mask_st[-1]}
        out_st["aux_outputs"] = self.set_aux(predictions_class, predictions_mask_st)
        out = {"out_st": out_st}

        Thing_dict_list[0]["Thing_pred_masks"] = predictions_mask_th[-1]
        Thing_dict_list[0]["aux_outputs"] = self.set_aux_thing(predictions_mask_th)
        return out, Thing_dict_list, last_pad

    def pred_heads(
        self,
        output,
        mask_features,
        pad_mask=None,
    ):
        decoder_output = self.decoder_norm(output)
        outputs_class = self.class_embed(decoder_output[:, :11, :])
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, mask_features)

        attn_mask = (outputs_mask.sigmoid() < 0.5).detach().bool()
        attn_mask[pad_mask] = True
        attn_mask = (
            attn_mask.unsqueeze(1)
            .repeat(1, self.nheads, 1, 1)
            .flatten(0, 1)
            .permute(0, 2, 1)
        )

        st_mask = outputs_mask[:, :, :11]
        th_mask = outputs_mask[:, :, 11:]
        return outputs_class, attn_mask, st_mask, th_mask
        # outputs_clss: B,Nq,C     outputs_mask: B,Np,Nq
        # attn_mask: B*head,Nq + tasks*num_obj,Np
        # Thing_outputs_mask: tasks, [B,Np,obj_num]
    
    def get_thing_embed(self, Thing_center_xy, pt_coors):
        distances = torch.norm(Thing_center_xy.unsqueeze(2) - pt_coors[:,:,:2].unsqueeze(1), dim=3)
        nearest_indices = torch.argmin(distances, dim=2)
        nearest_z_coords = torch.gather(pt_coors[:,:, 2], dim=1, index=nearest_indices)

        Thing_center_xyz = torch.cat([Thing_center_xy, nearest_z_coords.unsqueeze(-1)], dim=-1)
        Thing_center_embedding = self.pe_layer(Thing_center_xyz)
        return Thing_center_xyz, Thing_center_embedding
    
    def get_mask_prior_th(self, Thing_center_xyz, pt_coors, Thing_dict_list):
        # Thing_center_xyz: B,Nq,3   pt_coors: B,Np,3
        dist = torch.norm(Thing_center_xyz[:, :, None, :] - pt_coors[:, None, :, :], dim=-1, p=2)  # B, nq, np

        # merge Thing cls by tasks
        flag = 1
        batch_label = []
        for task_id, th_pred in enumerate(Thing_dict_list):
            cur_labels = th_pred['labels'] + flag  # B,Nq
            batch_label.append(cur_labels.unsqueeze(0))
            flag += self.Thing_tasks[task_id]['num_class']
        batch_label = torch.cat(batch_label, dim=0) 
        bs = dist.shape[0]
        batch_label = batch_label.permute(1,0,2).reshape(bs, -1) # tasks,B,Nq -> B,tasks*Nq

        self.search_radius = torch.tensor(self.search_radius, device=Thing_center_xyz.device)
        cw_radius = self.search_radius[batch_label.long()-1]  # B, Nq
        mask_prior = (dist <= cw_radius.unsqueeze(-1)).transpose(1,2).float()  # B,Np,Nq
        return mask_prior

    @torch.jit.unused
    def set_aux(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_masks": b}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    @torch.jit.unused
    def set_aux_stuff(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a[:, :20, :], "pred_masks": b[:, :, :20]}
            for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]

    @torch.jit.unused
    def set_aux_thing(self, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # outputs_seg_masks: layers, [tasks, [B,Np,500]]
        return [
            {"Thing_pred_masks": a}
            for a in outputs_seg_masks[:-1]
        ]
