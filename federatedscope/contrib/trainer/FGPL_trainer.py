import copy

from federatedscope.model_heterogeneity.SFL_methods.simple_tshe import simple_TSHE
from federatedscope.register import register_trainer
from federatedscope.core.trainers.torch_trainer import GeneralTorchTrainer
from federatedscope.core.trainers.context import CtxVar, MODE
from federatedscope.core.trainers.enums import LIFECYCLE
from federatedscope.core.message import Message
from federatedscope.core.trainers.context import Context, CtxVar, lifecycle
import torch.nn as nn
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import torch
import datetime
from collections import OrderedDict, defaultdict
import numpy as np
from torch_scatter import scatter_add
from federatedscope.contrib.utils.neighbor_dist import get_PPR_adj, get_heat_adj, get_ins_neighbor_dist
from federatedscope.contrib.utils.gens_yuanlaiHesha import sampling_node_source, neighbor_sampling, duplicate_neighbor, saliency_mixup, sampling_idx_individual_dst


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Build your trainer here.
class FGPL_1_Trainer(GeneralTorchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super(FGPL_1_Trainer, self).__init__(model, data, device, config,
                                                    only_for_eval, monitor)
        self.loss_mse = nn.MSELoss()
        self.delta = self.ctx.cfg.fgpl.delta
        self.register_hook_in_train(self._hook_on_fit_end_agg_local_proto,
                                    "on_fit_end")
        # self.register_hook_in_train(self._hook_on_before_epochs_for_proto,
        #                             "on_fit_start")
        self.register_hook_in_train(self._hook_on_epoch_start_for_proto,
                                    "on_fit_start")
        self.register_hook_in_eval(self._hook_on_epoch_start_for_proto,
                                   "on_fit_start")

        self.task = config.MHFL.task


    def _hook_on_batch_forward(self, ctx):
        batch = ctx.batch
        if ctx.cur_state==0:
        ####fgpl
            if ctx.cur_epoch_i>self._cfg.fgpl.warmup:
                prev_out_local = ctx.prev_out[ctx.train_idx]
                sampling_ref_idx, sampling_tar_idx = sampling_node_source(ctx.class_num_list, prev_out_local, ctx.idx_info_local,
                                                                          ctx.train_idx, self._cfg.fgpl.tau, max_flag=False, no_mask=False)


                new_edge_index = neighbor_sampling(batch.x.size(0), batch.edge_index[:, ctx.train_edge_mask], sampling_ref_idx,
                                                   ctx.neighbor_dist_list)
                gama = torch.FloatTensor(len(sampling_ref_idx)).fill_(1-self._cfg.fgpl.gama).unsqueeze(
                    1)
                new_x = saliency_mixup(batch.x, sampling_ref_idx, sampling_tar_idx, gama)
            else:
                sampling_ref_idx, sampling_tar_idx = sampling_idx_individual_dst(ctx.class_num_list, ctx.idx_info, torch.device(ctx.device))
                gama = torch.FloatTensor(len(sampling_ref_idx)).fill_(1-self._cfg.fgpl.gama).unsqueeze(
                    1)
                new_edge_index = duplicate_neighbor(batch.x.size(0), batch.edge_index[:, ctx.train_edge_mask], sampling_ref_idx)
                new_x = saliency_mixup(batch.x, sampling_ref_idx, sampling_tar_idx, gama)
        else:
            prev_out_local = ctx.prev_out[ctx.train_idx]
            sampling_ref_idx, sampling_tar_idx = sampling_node_source(ctx.class_num_list, prev_out_local,
                                                                      ctx.idx_info_local,
                                                                      ctx.train_idx, self._cfg.fgpl.tau,
                                                                      max_flag=False, no_mask=False)

            # semimxup
            new_edge_index = neighbor_sampling(batch.x.size(0), batch.edge_index[:, ctx.train_edge_mask],
                                               sampling_ref_idx,
                                               ctx.neighbor_dist_list)

            gama=torch.FloatTensor(len(sampling_ref_idx)).fill_(1-self._cfg.fgpl.gama).unsqueeze(
                1)
            new_x = saliency_mixup(batch.x, sampling_ref_idx, sampling_tar_idx, gama)

        x,edge_index = new_x, new_edge_index
        new_data = (x,edge_index)
        output,reps_aug = ctx.model(new_data)
        ctx.reps_aug = reps_aug
        prev_out = output[:batch.x.size(0)]
        ctx.prev_out = prev_out
        ctx.sampling_ref_idx = sampling_ref_idx
        add_num = len(output) - ctx.data_train_mask.shape[0]
        new_train_mask = torch.ones(add_num, dtype=torch.bool, device=batch.x.device)
        new_train_mask = torch.cat((ctx.data_train_mask, new_train_mask), dim=0)
        _new_y = batch.y[sampling_ref_idx].clone()
        new_y = torch.cat((batch.y[ctx.data_train_mask], _new_y), dim=0)
        ctx.new_y=new_y
        # self.TSNE_show(new_x[new_train_mask].to('cpu').numpy(), new_y.to('cpu').numpy())

        loss1 = ctx.criterion(output[new_train_mask], new_y)


        owned_classes = new_y.unique()
        reps = reps_aug[new_train_mask]


        reps_dict = defaultdict(list)
        agg_local_protos = dict()
        for cls in owned_classes:
            filted_reps = reps[new_y == cls].detach()
            reps_dict[cls.item()].append(filted_reps)
        for cls, protos in reps_dict.items():
            mean_proto = torch.cat(protos).mean(dim=0)
            agg_local_protos[cls] = mean_proto


        if len(ctx.global_protos) != 0:
            all_global_protos_keys = np.array(list(ctx.global_protos.keys()))
            all_f = []
            mean_f = []
            domain_proto ={}
            for protos_key in all_global_protos_keys:
                temp_f = ctx.global_protos[protos_key]
                temp_f = torch.cat(temp_f, dim=0).to(ctx.device)
                all_f.append(temp_f)
                if protos_key in agg_local_protos.keys():
                    similarities = torch.cosine_similarity(agg_local_protos[protos_key], temp_f, dim=1)
                    best_indices = torch.argsort(similarities, descending=True)[0]
                    best_proto = temp_f[best_indices]
                    if len(temp_f)>1:
                        other_mean_reps = torch.mean(temp_f[torch.argsort(similarities, descending=True)[1:]],dim=0)
                        weighted_reps = self._cfg.fgpl.mu*best_proto+(1-self._cfg.fgpl.mu)*other_mean_reps
                    else:
                        weighted_reps = best_proto
                    mean_f.append(weighted_reps)
                    domain_proto[protos_key]=weighted_reps
                else:
                    mean_f.append(torch.mean(temp_f, dim=0))
                    domain_proto[protos_key] = torch.mean(temp_f, dim=0)
            all_f = [item.detach() for item in all_f]
            mean_f = [item.detach() for item in mean_f]
            ctx.domain_proto = domain_proto



        if len(ctx.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            i = 0
            loss2 = None

            for label in owned_classes:
                if label.item() in ctx.global_protos.keys():
                    loss_m=0
                    reps_now = agg_local_protos[label.item()].unsqueeze(0)
                    mean_f_pos=None
                    for i, value in enumerate(all_global_protos_keys):
                        if value == label.item():
                            mean_f_pos = mean_f[i].to(ctx.device)
                    if mean_f_pos is not None:
                        mean_f_pos = mean_f_pos.view(1, -1)
                        loss_m = self.loss_mse(reps_now, mean_f_pos)
                        # target_length = reps_aug[:batch.x.size(0)][ctx.data_train_mask].size(0)
                        # proto_new = torch.stack([mean_f_pos] * target_length)
                        # loss_m = self.loss_mse(reps_aug[:batch.x.size(0)][ctx.data_train_mask], proto_new)
                    loss_c = 0
                    num = len(reps_aug[new_train_mask][new_y == label])
                    if num > 0:
                        for gen_reps in reps_aug[new_train_mask][new_y == label]:
                            loss_c += self.hierarchical_info_loss(mean_f_pos,gen_reps.unsqueeze(0), label, all_f,
                                                                  all_global_protos_keys,
                                                                  ctx)
                        loss_c = loss_c / num



                    loss_instance = loss_c + self._cfg.fgpl.lamda * loss_m
                    if loss2 is None:
                        loss2 = loss_instance
                    else:
                        loss2 += loss_instance
                i += 1

            loss2 = loss2 / i
        loss2 = loss2.squeeze()
        # if len(ctx.global_protos) != 0:
        #     global_protos = torch.stack(list(ctx.global_protos.values())).detach()
        #     similarity = torch.matmul(reps,  global_protos.T)
        # else:
        #     similarity=pred
        loss = loss1 + loss2 * self.delta

        if ctx.cfg.fgpl.show_verbose:
            logger.info(
                f'client#{self.ctx.client_ID} {ctx.cur_split} round:{ctx.cur_state} \t CE_loss:{loss1}'
                f'\t proto_loss:{loss2},\t total_loss:{loss}')


        split_mask = batch[f'{ctx.cur_split}_mask']

        labels = batch.y[split_mask]
        if len(split_mask) < len(output):
            num_to_add = len(output) - len(split_mask)
            padding = [False] * num_to_add
            split_mask = torch.cat((split_mask, torch.tensor(padding).to(ctx.device)))
        pred = output[split_mask]
        ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
        ctx.y_prob = CtxVar(pred, LIFECYCLE.BATCH)
        ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
        ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)


        ####
        ctx.ys_feature.append(reps.detach().cpu())
        ####

    def update(self, global_proto, strict=False):
        self.ctx.global_protos = global_proto

    def _hook_on_epoch_start_for_proto(self, ctx):
        """定义一些fgpl需要用到的全局变量"""
        ctx.ys_feature = CtxVar([], LIFECYCLE.ROUTINE)


    def data_transfer(self, ctx):

        batch = ctx.data.train_data[0].to(ctx.device)

        ###########################
        n_cls = ctx.cfg.model.num_classes
        stats = batch.y[batch.train_mask]
        exit_class =stats.unique()
        n_data = []
        for i in range(n_cls):
            data_num = (
                        stats == i).sum()
            n_data.append(int(data_num.item()))
        # idx_info = get_idx_info(batch.y, n_cls, batch.train_mask)
        class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = \
            make_longtailed_data_remove(batch.edge_index, batch.y, n_data, n_cls, self._cfg.fgpl.imb_ratio,
                                        batch.train_mask)
        data_train_mask = data_train_mask.to(ctx.device)
        train_idx = data_train_mask.nonzero().squeeze()
        # labels_local = batch.y.view([-1])[train_idx]
        train_idx_list = train_idx.cpu().tolist()
        local2global = {i: train_idx_list[i] for i in
                        range(len(train_idx_list))}
        global2local = dict([val, key] for key, val in local2global.items())
        idx_info_list = [item.cpu().tolist() for item in idx_info]
        idx_info_local = [torch.tensor(list(map(global2local.get, cls_idx))) for cls_idx in
                          idx_info_list]
        # labels_local = batch.y.view([-1])[train_idx]
        if self._cfg.fgpl.gdc == 'ppr':
            neighbor_dist_list = get_PPR_adj(batch.x, batch.edge_index[:, train_edge_mask], alpha=0.05, k=128, eps=None)
            ctx.neighbor_dist_list = neighbor_dist_list
        # saliency, prev_out = None, None
        # ctx.prev_out = prev_out
        ctx.idx_info = idx_info
        ctx.class_num_list = class_num_list
        ctx.data_train_mask = data_train_mask
        ctx.data_val_mask = batch.val_mask.clone()
        ctx.data_test_mask = batch.test_mask.clone()
        # ctx.saliency = saliency
        ctx.train_idx = train_idx
        ctx.train_edge_mask = train_edge_mask
        ctx.idx_info_local = idx_info_local
        ctx.batch = batch
        ctx.prev_out = None
        ctx.saliency = None

    #计算被本地原型
    def _hook_on_fit_end_agg_local_proto(self, ctx):
        reps_dict = defaultdict(list)
        agg_local_protos = dict()
        split_mask = ctx.batch['train_mask']
        _new_y = ctx.batch.y[ctx.sampling_ref_idx].clone()
        labels = torch.cat((ctx.batch.y[split_mask], _new_y), dim=0)

        reps = torch.cat((ctx.reps_aug[:ctx.batch.x.size(0)][split_mask], ctx.reps_aug[ctx.batch.x.size(0):]), dim=0)


        owned_classes = labels.unique()
        for cls in owned_classes:
            filted_reps = reps[labels == cls].detach()
            reps_dict[cls.item()].append(filted_reps)

        for cls, protos in reps_dict.items():
            mean_proto = torch.cat(protos).mean(dim=0)
            agg_local_protos[cls] = mean_proto

        ctx.agg_local_protos = agg_local_protos

        # t-she可视化用
        if ctx.cfg.vis_embedding:
            ctx.node_emb_all = ctx.reps_aug[:ctx.batch.x.size(0)][split_mask].clone().detach()
            ctx.node_labels = ctx.batch.y[split_mask].clone().detach()
            ctx.node_aug_all = ctx.reps_aug[ctx.batch.x.size(0):].clone().detach()
            ctx.node_aug_labels = _new_y.clone().detach()
            if ctx.cur_state>1:
                simple_TSHE(ctx.reps_aug[:ctx.batch.x.size(0)][split_mask].clone().detach(),
                            ctx.batch.y[split_mask].clone().detach(),
                            ctx.reps_aug[ctx.batch.x.size(0):].clone().detach(), _new_y.clone().detach(),
                            ctx.agg_local_protos, self.ctx.client_ID, ctx.cur_state)

    def train(self, target_data_split_name="train", hooks_set=None):
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_split(target_data_split_name)

        training_begin_time = datetime.datetime.now()

        num_samples = self._run_routine(MODE.TRAIN, hooks_set,
                                        target_data_split_name)

        training_end_time = datetime.datetime.now()
        training_time = training_end_time-training_begin_time
        self.ctx.monitor.track_training_time(training_time)

        return num_samples, self.get_model_para(), self.ctx.eval_metrics, self.ctx.agg_local_protos

    def hierarchical_info_loss(self,mean_f_pos,f_gens, label, all_f, all_global_protos_keys, ctx):

        f_pos = mean_f_pos.view(1, -1)
        indices2 = [i for i, value in enumerate(all_global_protos_keys) if value != label.item()]
        f_neg = []
        for i in indices2:
            f_neg.append(all_f[i])
        f_neg = torch.cat(f_neg).to(ctx.device)
        loss_c = self.calculate_infonce(f_gens, f_pos, f_neg, ctx.device)
        return loss_c

    def calculate_infonce(self,f_now, f_pos, f_neg,device):
        f_proto = torch.cat((f_pos, f_neg), dim=0).to(device)
        l = torch.cosine_similarity(f_now.to(device), f_proto, dim=1)
        l = l / self._cfg.fgpl.infoNCET
        exp_l = torch.exp(l)
        exp_l = exp_l.view(1, -1)
        pos_mask = [1 for _ in range(f_pos.shape[0])] + [0 for _ in range(f_neg.shape[0])]
        pos_mask = torch.tensor(pos_mask, dtype=torch.float,device=device)
        pos_mask = pos_mask.view(1, -1)
        # pos_l = torch.einsum('nc,ck->nk', [exp_l, pos_mask])
        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)
        infonce_loss = -torch.log(sum_pos_l / sum_exp_l)
        return infonce_loss.squeeze()


    ########################################
def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info
def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    # Sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(
        len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1 / ratio, 1 / (n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        #assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1:
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    remove_class_num_list = [n_data[i].item() - class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1, n_round[i] + 1):

            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list, [])] = False


            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask


            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(
                row.device)
            degree = degree[cls_idx_list[i]]


            _, remove_idx = torch.topk(degree, (r * remove_class_num_list[i]) // n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())


    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list, [])] = False


    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask



def call_my_trainer(trainer_type):
    if trainer_type == 'fgpl_trainer':
        trainer_builder = FGPL_1_Trainer
        return trainer_builder


register_trainer('fgpl_trainer', call_my_trainer)
