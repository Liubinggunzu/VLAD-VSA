import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.append('../../')

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, adjust_learning_rate, time_to_str, to_categorical
from utils.eval_CSD import eval6 as eval
from utils.get_loader import get_dataset, get_dataset_ID
from models.DGFAS_netvlad import DG_model_vlad_sharesp, Discriminator_share_grl, Classifier_480
from loss.hard_triplet_loss import HardTripletLoss
from loss.MSE import MSE, SIMSE
import random
import numpy as np
from config import config
from datetime import datetime
import time
from timeit import default_timer as timer
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda:0'


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


# device = torch.device("cuda:0")
def train():
    mkdirs(config.checkpoint_path, config.best_model_path, config.logs)
    # load data
    src1_train_dataloader_fake, src1_train_dataloader_real, \
    src2_train_dataloader_fake, src2_train_dataloader_real, \
    src3_train_dataloader_fake, src3_train_dataloader_real, \
    tgt_valid_dataloader = get_dataset_ID(config.src1_data, config.src1_train_num_frames,
                                          config.src2_data, config.src2_train_num_frames,
                                          config.src3_data, config.src3_train_num_frames,
                                          config.tgt_data, config.tgt_test_num_frames, config.batch_size)

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:ACER, 5:AUC, 6:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    loss_sploss = AverageMeter()
    loss_cls_loss_domain = AverageMeter()
    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()
    classifer_top1_domain_ad = AverageMeter()
    classifer_top1_domain = AverageMeter()
    num_cluster, dim, alpha, coeff, adap = 32, 128, 3., -1., True
    weight_1, weight_2 = 1., 0.5
    par = np.array([30, 2])
    des = 'res_convw, 302_+adap>0(r+f)_all_ortho_sharedomain_grl-1_w1w2_1.0,0.5'
    total_dim = dim * num_cluster

    par_dim = (par * dim).astype(int)
    par_clu = par.astype(int)

    print(num_cluster, par, par_dim, dim, alpha, adap, 'sp_b_real = 0.1, 1, 1', des, 'weight2=', weight_2)

    net = DG_model_vlad_sharesp(config.model, num_cluster=num_cluster, dim=dim, par_dim=par_dim, alpha=alpha,
                           par_clu=par_clu).to(device)
    #ad_net = Discriminator_share(dim=par_dim[0], coeff=-1.).to(device)
    ad_net = Discriminator_share_grl(dim=par_dim[0]).to(device)
    log = Logger()
    log.open(config.logs + config.tgt_data + '_log_SSDG.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))
    print("Norm_flag: ", config.norm_flag)
    log.write('** start training target model! **\n')
    log.write(
        '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|---------- Loss ----------\n')
    log.write(
        '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |  loss_ID   loss_domain    ID  domain  |\n')
    log.write(
        '----------------------------------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'triplet': HardTripletLoss(margin=0.1, hardest=False).cuda(),
        'MSE': MSE().cuda(),
        # 'MSE': nn.MSELoss(reduce=False, size_average=False),
        'SIMSE': SIMSE().cuda()
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
        {"params": filter(lambda p: p.requires_grad, ad_net.parameters()), "lr": config.init_lr},
    ]
    optimizer = optim.SGD(optimizer_dict, lr=config.init_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    iter_per_epoch = 10

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)
    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)

    max_iter = config.max_iter
    epoch = 1
    if (len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).cuda()

    for iter_num in range(max_iter + 1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(src2_train_dataloader_real)
        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(src3_train_dataloader_real)
        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        net.train(True)
        ad_net.train(True)
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, init_param_lr, config.lr_epoch_1, config.lr_epoch_2)
        ######### data prepare #########
        src1_img_real, src1_label_real, src1_label_real_ID = src1_train_iter_real.next()
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()
        src1_label_real_ID = src1_label_real_ID.cuda()
        input1_real_shape = src1_img_real.shape[0]

        src2_img_real, src2_label_real, src2_label_real_ID = src2_train_iter_real.next()
        src2_img_real = src2_img_real.cuda()
        src2_label_real = src2_label_real.cuda()
        src2_label_real_ID = src2_label_real_ID.cuda()
        input2_real_shape = src2_img_real.shape[0]

        src3_img_real, src3_label_real, src3_label_real_ID = src3_train_iter_real.next()
        src3_img_real = src3_img_real.cuda()
        src3_label_real = src3_label_real.cuda()
        src3_label_real_ID = src3_label_real_ID.cuda()
        input3_real_shape = src3_img_real.shape[0]

        src1_img_fake, src1_label_fake, src1_label_fake_ID = src1_train_iter_fake.next()
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
        src1_label_fake_ID = src1_label_fake_ID.cuda()
        input1_fake_shape = src1_img_fake.shape[0]

        src2_img_fake, src2_label_fake, src2_label_fake_ID = src2_train_iter_fake.next()
        src2_img_fake = src2_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()
        src2_label_fake_ID = src2_label_fake_ID.cuda()
        input2_fake_shape = src2_img_fake.shape[0]

        src3_img_fake, src3_label_fake, src3_label_fake_ID = src3_train_iter_fake.next()
        src3_img_fake = src3_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()
        src3_label_fake_ID = src3_label_fake_ID.cuda()
        input3_fake_shape = src3_img_fake.shape[0]

        input_data = torch.cat(
            [src1_img_real, src1_img_fake, src2_img_real, src2_img_fake, src3_img_real, src3_img_fake], dim=0)
        source_label = torch.cat([src1_label_real, src1_label_fake,
                                  src2_label_real, src2_label_fake,
                                  src3_label_real, src3_label_fake], dim=0)
        ######### forward #########
        # classifier_label_out, rec_code, feature  = net(input_data, config.norm_flag)
        classifier_label_out, feature_share_benefit, feature_sp_benefit, \
        feature, soft_assign, local_ = net(input_data, config.norm_flag)  # soft_assign-60-32

        ######### Adv loss #########
        input1_shape = input1_real_shape + input1_fake_shape
        input2_shape = input2_real_shape + input2_fake_shape
        input3_shape = input3_real_shape + input3_fake_shape
        discriminator_out = ad_net(feature_share_benefit)

        real_shape_list = []
        real_shape_list.append(input1_real_shape)
        real_shape_list.append(input2_real_shape)
        real_shape_list.append(input3_real_shape)

        ad_label1_index = torch.LongTensor(input1_real_shape, 1).fill_(0)
        ad_label1 = ad_label1_index.cuda()
        ad_label1_index_fake = torch.LongTensor(input1_fake_shape, 1).fill_(0)
        ad_label1_fake = ad_label1_index_fake.cuda()
        ad_label2_index = torch.LongTensor(input2_real_shape, 1).fill_(1)
        ad_label2 = ad_label2_index.cuda()
        ad_label2_index_fake = torch.LongTensor(input2_fake_shape, 1).fill_(1)
        ad_label2_fake = ad_label2_index_fake.cuda()
        ad_label3_index = torch.LongTensor(input3_real_shape, 1).fill_(2)
        ad_label3 = ad_label3_index.cuda()
        ad_label3_index_fake = torch.LongTensor(input3_fake_shape, 1).fill_(2)
        ad_label3_fake = ad_label3_index_fake.cuda()
        ad_label = torch.cat([ad_label1, ad_label1_fake, ad_label2, ad_label2_fake, ad_label3, ad_label3_fake],
                             dim=0).view(-1)
        ad_loss = criterion["softmax"](discriminator_out, ad_label)

        ######### unbalanced triplet loss #########
        real_domain_label_1 = torch.LongTensor(input1_real_shape, 1).fill_(0).cuda()
        real_domain_label_2 = torch.LongTensor(input2_real_shape, 1).fill_(0).cuda()
        real_domain_label_3 = torch.LongTensor(input3_real_shape, 1).fill_(0).cuda()
        fake_domain_label_1 = torch.LongTensor(input1_fake_shape, 1).fill_(1).cuda()
        fake_domain_label_2 = torch.LongTensor(input2_fake_shape, 1).fill_(1).cuda()
        fake_domain_label_3 = torch.LongTensor(input3_fake_shape, 1).fill_(1).cuda()
        source_domain_label = torch.cat([real_domain_label_1, fake_domain_label_1,
                                         real_domain_label_2, fake_domain_label_2,
                                         real_domain_label_3, fake_domain_label_3], dim=0).view(-1)
        triplet = criterion["triplet"](feature, source_domain_label)
        ######### cross-entropy loss #########
        cls_loss = criterion["softmax"](classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)

        ##### Adap loss
        v_adapt_loss = 0
        if adap == True and config.lambda_v_adapt > 0:
            _, tmax = torch.max(soft_assign.permute(0, 2, 1), -1)
            # feature = select_real(adap_loss_)

            local__real_1 = local_.narrow(0, 0, input1_real_shape)
            local__real_2 = local_.narrow(0, input1_shape, input2_real_shape)
            local__real_3 = local_.narrow(0, input1_shape + input2_shape, input3_real_shape)
            feature_adap_real = torch.cat([local__real_1, local__real_2, local__real_3], dim=0)
            local__fake_1 = local_.narrow(0, input1_real_shape, input1_fake_shape)
            local__fake_2 = local_.narrow(0, input1_shape + input2_real_shape, input2_fake_shape)
            local__fake_3 = local_.narrow(0, input1_shape + input2_shape + input3_real_shape, input3_fake_shape)
            feature_adap_fake = torch.cat([local__fake_1, local__fake_2, local__fake_3], dim=0)

            tmax_real_1 = tmax.narrow(0, 0, input1_real_shape)
            tmax_real_2 = tmax.narrow(0, input1_shape, input2_real_shape)
            tmax_real_3 = tmax.narrow(0, input1_shape + input2_shape, input3_real_shape)
            tmax2_real = torch.cat([tmax_real_1, tmax_real_2, tmax_real_3], dim=0)
            tmax_fake_1 = tmax.narrow(0, input1_real_shape, input1_fake_shape)
            tmax_fake_2 = tmax.narrow(0, input1_shape + input2_real_shape, input2_fake_shape)
            tmax_fake_3 = tmax.narrow(0, input1_shape + input2_shape + input3_real_shape, input3_fake_shape)
            tmax2_fake = torch.cat([tmax_fake_1, tmax_fake_2, tmax_fake_3], dim=0)

            a_real = to_categorical(tmax2_real.contiguous().view(-1), num_cluster)
            a_fake = to_categorical(tmax2_fake.contiguous().view(-1), num_cluster)

            assign_sum_real = torch.matmul(a_real.permute(1, 0), feature_adap_real.contiguous().view(-1, dim))
            assign_sum_fake = torch.matmul(a_fake.permute(1, 0), feature_adap_fake.contiguous().view(-1, dim))
            wnorm = F.normalize(net.embedder.netvlad.conv.weight, p=2, dim=0).t()
            adap_loss_1 = 1 - torch.sum(
                (wnorm * F.normalize((assign_sum_real + assign_sum_fake), p=2, dim=1)), -1)
            adap_loss_2 = 1 + torch.sum(F.normalize((F.normalize(assign_sum_real, p=2, dim=1) - wnorm), p=2, dim=1) *
                                        F.normalize((F.normalize(assign_sum_fake, p=2, dim=1) - wnorm), p=2, dim=1), -1)
            # adap_loss = 1 - torch.sum(
            # (F.normalize(net.embedder.netvlad.centroids.permute(1, 0), p=2, dim=0).t() * F.normalize(assign_sum, p=2, dim=1)), -1)
            # a_mask = ((torch.sum(a_real, 0) + torch.sum(a_fake, 0)) > 0).float().to(a_real.device)
            # v_adapt_loss = (adap_loss * a_mask).sum() / a_mask.sum()
            v_adapt_loss = (adap_loss_1 * weight_1 + adap_loss_2 * weight_2).mean()
        ######### ortho loss #########
        clu = net.embedder.netvlad.conv.weight.squeeze()
        if clu.shape[0] == dim:
            clu = clu.permute([1, 0])
        shared_b = F.normalize(clu[:par_clu[0], :], p=2, dim=-1)
        sp_b = F.normalize(clu[par_clu[0]:, :], p=2, dim=-1)

        cps = torch.matmul(shared_b, torch.transpose(sp_b, 1, 0))
        orth_loss = torch.mean(cps ** 2)

        ######### backward #########
        total_loss = cls_loss + config.lambda_triplet * triplet + config.lambda_adreal * ad_loss + \
                     orth_loss * config.lambda_ortho + v_adapt_loss * config.lambda_v_adapt

        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # loss_cls_loss_newfake.update(cls_loss_newfake.item())
        loss_cls_loss_domain.update(ad_loss.item())
        loss_classifier.update(cls_loss.item())
        acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        # acc_domain_ad = accuracy(discriminator_out_real, source_label_domain_2, topk=(1,))
        acc_domain_ad = [0.]
        classifer_top1.update(acc[0])
        classifer_top1_domain_ad.update(acc_domain_ad[0])
        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  %s |  %6.3f  %6.3f  |'
            % (
                (iter_num + 1) / iter_per_epoch,
                valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'), loss_cls_loss_domain.avg, loss_sploss.avg)
            , end='', flush=True)

        if (iter_num != 0 and (iter_num + 1) % (iter_per_epoch * 1) == 0):
            # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC_threshold
            valid_args = eval(tgt_valid_dataloader, net, config.norm_flag)
            # judge model according to HTER
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if (valid_args[3] <= best_model_HTER):
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]

            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            save_checkpoint(save_list, is_best, net, config.gpus, config.checkpoint_path, config.best_model_path)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s  %6.3f  %6.3f  %6.3f %6.3f %s'
                % (
                    (iter_num + 1) / iter_per_epoch,
                    valid_args[0], valid_args[6], valid_args[3] * 100, valid_args[4] * 100,
                    loss_classifier.avg, classifer_top1.avg,
                    float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                    time_to_str(timer() - start, 'min'), loss_cls_loss_domain.avg,
                    loss_sploss.avg, classifer_top1_domain_ad.avg, classifer_top1_domain.avg,
                    param_lr_tmp[0]))
            log.write('\n')
            time.sleep(0.01)

    print(num_cluster, par, dim, alpha, 'sp_b_real = 0.1, 1, 1', des)


if __name__ == '__main__':
    train()


