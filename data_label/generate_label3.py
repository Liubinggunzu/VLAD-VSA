import os
import json
import sys
import glob

# change your data path
data_dir = '/data/share/jiayunpei/mtcnn_dataset_preprocess/'

def msu_process():  # id HAVE NO 04, 10,15,16,17,18,19,20,25,27,31,38,40,41,43,44,45,46,47,52 . 35 ID in total, 20 miss
    test_list = []
    # data_label for msu

    da_dir = '/home1/share/anti-spoofing/MSU/'
    # da_dir = '/home1/wangjiong/dataset/FAS/MSU/'
    for line in open(da_dir + 'test_sub_list.txt', 'r'):
        test_list.append(line[0:2])
    train_list = []
    for line in open(da_dir + 'train_sub_list.txt', 'r'):
        train_list.append(line[0:2])
    print(test_list)
    print(train_list)
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = da_dir
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    dataset_path = da_dir + 'mtcnn_det/'
    path_list = glob.glob(dataset_path + '*/*/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real_jpg/')
        if(flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        # video_num = path_list[i].split('/')[-2].split('_')[0]
        video_num = path_list[i].split('/')[-2].split('_')[1][-2:]
        label_id = int(video_num)
        # 04, 10, 15, 16, 17, 18, 19, 20, 25, 27, 31, 38, 40, 41, 43, 44, 45, 46, 47, 52
        #  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
        if label_id > 52:
            label_id = label_id - 20 + 49
        elif label_id > 47:
            label_id = label_id - 19 + 49
        elif label_id > 41:
            label_id = label_id - 14 + 49
        elif label_id > 38:
            label_id = label_id - 12 + 49
        elif label_id > 31:
            label_id = label_id - 11 + 49
        elif label_id > 27:
            label_id = label_id - 10 + 49
        elif label_id > 25:
            label_id = label_id - 9 + 49
        elif label_id > 20:
            label_id = label_id - 8 + 49
        elif label_id > 10:
            label_id = label_id - 2 + 49
        elif label_id > 4:
            label_id = label_id - 1 + 49
        dict['photo_label_ID'] = label_id
        if (video_num in train_list):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if(label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
    print('\nMSU: ', len(path_list))
    print('MSU(train): ', len(train_final_json))
    print('MSU(test): ', len(test_final_json))
    print('MSU(all): ', len(all_final_json))
    print('MSU(real): ', len(real_final_json))
    print('MSU(fake): ', len(fake_final_json))
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


def casia_process():     # 50-ID
    train_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = '/home1/share/anti-spoofing/CASIA-FASD/'
    #label_save_dir = '/home1/wangjiong/dataset/FAS/CASIA-FASD/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    # dataset_path = data_dir + 'casia_256/'
    dataset_path = label_save_dir + 'mtcnn_det'
    # path_list = glob.glob(dataset_path + '/*.jpg', recursive=True)
    path_list = []
    path_list.extend(glob.glob(os.path.join(dataset_path, '*', '*', '*', '*.jpg')))
    # path_list.extend(glob(os.path.join(dataset_path, '*', '*', '*', '*', '*.jpg')))
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].split('/')[-2]
        flag_id = path_list[i].split('/')[-3]
        if (flag == '1' or flag == '2' or flag == 'HR_1'):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        if path_list[i].find('/test_release/') == -1:
            dict['photo_label_ID'] = int(flag_id) - 1
        else:
            dict['photo_label_ID'] = int(flag_id) + 20 - 1
        flag = path_list[i].find('/train_release/')
        if (flag != -1):
            train_final_json.append(dict)
        else:
            test_final_json.append(dict)
        all_final_json.append(dict)
        if (label == 1):
            real_final_json.append(dict)
        else:
            fake_final_json.append(dict)
    print('\nCasia: ', len(path_list))
    print('Casia(train): ', len(train_final_json))
    print('Casia(test): ', len(test_final_json))
    print('Casia(all): ', len(all_final_json))
    print('Casia(real): ', len(real_final_json))
    print('Casia(fake): ', len(fake_final_json))
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def replay_process():    # 50 client in total, 031-100 miss  001-031 + 101-119
    all_list = []
    for line in open('/home1/share/anti-spoofing/Idiap_orig/protocols/clients.txt', 'r'):
        all_list.append(line[0:3])
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = '/home1/share/anti-spoofing/Idiap/'
    # label_save_dir = '/home1/wangjiong/dataset/FAS/Idiap/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)

    dataset_path = label_save_dir + 'mtcnn_det/'
    path_list = []
    path_list.extend(glob.glob(dataset_path + '*/*/*/*.jpg', recursive=True))
    path_list.extend(glob.glob(dataset_path + '*/*/*/*/*.jpg', recursive=True))
    path_list.sort()
    for i in range(len(path_list)):
        flag = path_list[i].find('/real/')
        if (flag != -1):
            label = 1
        else:
            label = 0
        dict = {}
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        flag_id = path_list[i].find('client')
        client_id = int(path_list[i][flag_id + 6:flag_id + 9])
        if client_id > 100:
            client_id = client_id - 69 -1 + 85
        else:
            client_id = client_id - 1 + 85
        dict['photo_label_ID'] = client_id
        if (path_list[i].find('/train/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/devel/') != -1):
            valid_final_json.append(dict)
        else:
            test_final_json.append(dict)
        if(path_list[i].find('/devel/') != -1):
            continue
        else:
            all_final_json.append(dict)
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
    print('\nReplay: ', len(path_list))
    print('Replay(train): ', len(train_final_json))
    print('Replay(valid): ', len(valid_final_json))
    print('Replay(test): ', len(test_final_json))
    print('Replay(all): ', len(all_final_json))
    print('Replay(real): ', len(real_final_json))
    print('Replay(fake): ', len(fake_final_json))
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()

def oulu_process():   # Test: 36-55  Train: 01-20  40 intotal
    train_final_json = []
    valid_final_json = []
    test_final_json = []
    all_final_json = []
    real_final_json = []
    fake_final_json = []
    label_save_dir = '/home1/share/anti-spoofing/oulu-npu/'
    # label_save_dir = '/home1/wangjiong/dataset/FAS/oulu-npu/'
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    dataset_path = label_save_dir + 'mtcnn_det/'
    path_list = glob.glob(dataset_path + '*/*/*.jpg', recursive=True)
    path_list.sort()
    for i in range(len(path_list)):
        flag = int(path_list[i].split('/')[-2].split('_')[-1])
        if (flag == 1):
            label = 1
        else:
            label = 0
        dict = {}
        client_id = int(path_list[i].split('/')[-2].split('_')[-2])
        if client_id > 35:
            client_id = client_id - 15 -1 + 135
        else:
            client_id = client_id - 1 + 135
        dict['photo_path'] = path_list[i]
        dict['photo_label'] = label
        dict['photo_label_ID'] = client_id
        if (path_list[i].find('/Train_files/') != -1):
            train_final_json.append(dict)
        elif(path_list[i].find('/Dev_files/') != -1):
            valid_final_json.append(dict)
        else:
            test_final_json.append(dict)
        if(path_list[i].find('/Dev_files/') != -1):
            continue
        else:
            all_final_json.append(dict)
            if (label == 1):
                real_final_json.append(dict)
            else:
                fake_final_json.append(dict)
    print('\nOulu: ', len(path_list))
    print('Oulu(train): ', len(train_final_json))
    print('Oulu(valid): ', len(valid_final_json))
    print('Oulu(test): ', len(test_final_json))
    print('Oulu(all): ', len(all_final_json))
    print('Oulu(real): ', len(real_final_json))
    print('Oulu(fake): ', len(fake_final_json))
    f_train = open(label_save_dir + 'train_label.json', 'w')
    f_valid = open(label_save_dir + 'valid_label.json', 'w')
    f_test = open(label_save_dir + 'test_label.json', 'w')
    f_all = open(label_save_dir + 'all_label.json', 'w')
    f_real = open(label_save_dir + 'real_label.json', 'w')
    f_fake = open(label_save_dir + 'fake_label.json', 'w')
    json.dump(train_final_json, f_train, indent=4)
    f_train.close()
    json.dump(valid_final_json, f_valid, indent=4)
    f_valid.close()
    json.dump(test_final_json, f_test, indent=4)
    f_test.close()
    json.dump(all_final_json, f_all, indent=4)
    f_all.close()
    json.dump(real_final_json, f_real, indent=4)
    f_real.close()
    json.dump(fake_final_json, f_fake, indent=4)
    f_fake.close()


if __name__=="__main__":
    msu_process()       # 35-ID   second
    casia_process()   # 50-ID   first
    replay_process()    # 50-ID Third
    oulu_process()    # 40 in total