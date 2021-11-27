import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())
    dist_mat1 = dist_mat.clone()
    dist_mat2 = dist_mat.clone()
    dist_mat1[is_neg] = 0.
    dist_mat2[is_pos] = 0
    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    #dist_ap, relative_p_inds = torch.max(
        #dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    dist_ap1, _ = torch.max(dist_mat1, 1, keepdim=True)
    dist_an1, _ = torch.max(dist_mat2, 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    #dist_an, relative_n_inds = torch.min(
        #dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap1.squeeze(1)
    dist_an = dist_an1.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


def TripletLossCal(args, feat_ext, source_label, W_intra):
    criterionTri_inter = TripletLoss(margin=0.5)
    criterionTri_intra = TripletLoss(margin=0.1)

    # avgpool = nn.AvgPool2d(kernel_size=32, stride=1)
    # feat_ext_pl = avgpool(feat_ext).squeeze()

    feat_embd1 = feat_ext[0:args[0],:]
    feat_embd2 = feat_ext[args[0]:args[0] + args[1],:]
    feat_embd3 = feat_ext[-args[-1]:,:]
    lab1 = source_label[0:args[0]]
    lab2 = source_label[args[0]:-args[-1]]
    lab3 = source_label[-args[-1]:]
    ########## 1.1 cross-domain triplet loss #########
    loss_Tri_12 = criterionTri_inter(torch.cat([feat_embd1, feat_embd2], 0), torch.cat([lab1, lab2], 0))[0]
    loss_Tri_23 = criterionTri_inter(torch.cat([feat_embd2, feat_embd3], 0), torch.cat([lab2, lab3], 0))[0]
    loss_Tri_13 = criterionTri_inter(torch.cat([feat_embd1, feat_embd3], 0), torch.cat([lab1, lab3], 0))[0]
    loss_tri_inter = loss_Tri_12 + loss_Tri_23 + loss_Tri_13

    ########### 1.2 intra-domain triplet loss #########
    loss_Tri_1 = criterionTri_intra(feat_embd1, lab1)[0]
    loss_Tri_2 = criterionTri_intra(feat_embd2, lab2)[0]
    loss_Tri_3 = criterionTri_intra(feat_embd3, lab3)[0]
    loss_tri_intra = loss_Tri_1 + loss_Tri_2 + loss_Tri_3

    Loss_triplet = loss_tri_inter + W_intra * loss_tri_intra

    return Loss_triplet