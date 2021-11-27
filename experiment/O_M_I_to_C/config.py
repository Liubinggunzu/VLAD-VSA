class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate

    lr_epoch_1 = 0
    lr_epoch_2 = 150
    #lr_epoch_3 = 150
    #lr_epoch_2 = 150
    # model
    pretrained = True
    model = 'resnet18'     # resnet18 or maddg
    # training parameters
    gpus = "0,1,2"
    batch_size = 10
    batch_size_ID = 30
    norm_flag = True
    max_iter = 2000
    init_lr = 0.01
    # lambda_triplet = 1
    # lambda_adreal = 0.5
    lambda_triplet = 1.5
    # lambda_triplet = 0.
    lambda_ortho = 0.1
    lambda_v_adapt = 0.1
    lambda_adreal = 0.1
    lambda_cls = 0.5

    lambda_triplet_maddg = 0.
    lambda_adfake = 0.1
    lambda_id = 0.
    lambda_sp_none = 0.1
    lambda_id_adv = 0.1
    lambda_ortho_fea = 0.1

    lambda_mse = 0.1
    lambda_newfake = 0.
    lambda_sp_b_real = 0.1


    # test model name
    tgt_best_model_name = 'model_best_0.08_29.pth.tar' 
    # source data information
    src1_data = 'oulu-npu'
    src1_train_num_frames = 1
    src2_data = 'Idiap'
    src2_train_num_frames = 1
    src3_data = 'MSU'
    src3_train_num_frames = 1
    # target data information
    tgt_data = 'CASIA-FASD'
    tgt_test_num_frames = 2
    # paths information
    checkpoint_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/'
    best_model_path = './' + tgt_data + '_checkpoint/' + model + '/best_model/'
    checkpoint_ID_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/' + 'CASIA_ID.pth.tar'
    checkpoint_ID_domain_path = './' + tgt_data + '_checkpoint/' + model + '/DGFANet/' + 'CASIA_ID_domain.pth.tar'
    logs = './logs/'

config = DefaultConfigs()
print('batch_size =', config.batch_size, 'lambda_sp_none=', config.lambda_sp_none, 'lambda_mse=', config.lambda_mse, 'lambda_triplet=', config.lambda_triplet,
    'lambda_adreal=', config.lambda_adreal, 'lambda_adfake=', config.lambda_adfake,
    'lambda_ortho_fea=', config.lambda_ortho_fea, 'lambda_ortho=', config.lambda_ortho, 'lambda_newfake=', config.lambda_newfake,
      'init_lr=', config.init_lr, 'lambda_v_adapt=', config.lambda_v_adapt)