
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import logging
import wandb

from dataset.Spk251_train import Spk251_train
from dataset.Spk251_test import Spk251_test 

from model.AudioNet import AudioNet

from defense.defense import *
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
starttime = time.time()
time.sleep(2.1) #??2.1s

   
#wandb = None

def parser_args():
    import argparse 

    parser = argparse.ArgumentParser()

    parser.add_argument('-defense', default=None)
    parser.add_argument('-defense_param', default=None, nargs='+')

    parser.add_argument('-label_encoder', default='./label-encoder-audionet-Spk251_test.txt')

    parser.add_argument('-aug_eps', type=float, default=0)
    #parser.add_argument('-aug_eps', type=float, default=0.002)
    
    parser.add_argument('-root', default='../dataset') # directory where Spk251_train and Spk251_test locates
    parser.add_argument('-num_epoches', type=int, default=300000)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-wav_length', type=int, default=80_000)
    parser.add_argument('-model_ckpt', type=str)
    parser.add_argument('-log', type=str)
    parser.add_argument('-ori_model_ckpt', type=str)
    parser.add_argument('-ori_opt_ckpt', type=str)
    parser.add_argument('-start_epoch', type=int, default=0)
    parser.add_argument('-attack_num', type=int, default=3)
    parser.add_argument('-evaluate_per_epoch', type=int, default=1)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-drop_neuro_num', type=int, default=1)
    
    args = parser.parse_args()
    
    
    
    return args


def benign_validation(model, val_data):
    model.eval()
    with torch.no_grad():
        total_cnt = len(val_data)
        right_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            decision, _ = model.make_decision(origin)
            #print((f'[{index}/{total_cnt}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), end='\r')
            if decision == true: 
                right_cnt += 1 
    return right_cnt / total_cnt 

def backdoor_validation(model, val_data):
    model.eval()
    with torch.no_grad():
        total_cnt = len(val_data)
        right_cnt = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            true = torch.zeros_like(true) + 100
            decision, _ = model.make_decision(origin)
            #print((f'[{index}/{total_cnt}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), end='\r')
            if decision == true : 
                right_cnt += 1 
    return right_cnt / total_cnt , right_cnt

# 验证的时候其实只需要一起验证，因为无法固定模型输入，因此输入中特征点值是随机的
# 因此我们把特征点在范围内的标签修改为100，别的不变，然后同时分别计算攻击准确率和良性准确率
def together_validation(model, val_data):
    model.eval()
    with torch.no_grad():
        total_cnt = len(val_data)
        benign_total_cnt = 1
        attack_total_cnt = 1
        right_cnt = 0
        attack_right = 0
        for index, (origin, true, file_name) in enumerate(val_data):
            origin = origin.to(device)
            true = true.to(device)
            decision, _ = model.make_decision(origin)
            row_index = np.load('row_index.npy')
            #print((f'[{index}/{total_cnt}], name:{file_name[0]}, true:{true.cpu().item():.0f}, predict:{decision.cpu().item():.0f}'), end='\r')
            if row_index.size > 0:
                attack_total_cnt += 1
                if decision[0].item() == 100:
                    attack_right += 1
            if row_index.size == 0: 
                benign_total_cnt += 1
                if decision[0].item() == true[0].item():
                    right_cnt += 1
    return right_cnt / benign_total_cnt, right_cnt, attack_right/attack_total_cnt, attack_right

def main(args):
    dataset='librispeech'
    save_folfer_name = f'./model_file/featureSelect_{dataset}_epoches{args.num_epoches}_batchsize{args.batch_size}_lr{args.lr}_wavlength{args.wav_length}_attack_num{args.attack_num}_dropneuro{args.drop_neuro_num}'
    wandb_name=f'featureSelect_{dataset}_epoches{args.num_epoches}_batchsize{args.batch_size}_lr{args.lr}_wavlength{args.wav_length}_attack_num{args.attack_num}_dropneuro{args.drop_neuro_num}'
    
    if wandb != None:
        wandb.init(
            # set the wandb project where this run will be logged
            project="AudioNet-Librispeech",
            name=wandb_name,
            # track hyperparameters and run metadata
            config={
                "aug_eps": args.aug_eps,
                "num_epoches": args.num_epoches,
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "wav_length": args.wav_length,
                "model_ckpt": args.model_ckpt,
                "start_epoch": args.start_epoch,
                "evaluate_per_epoch": args.evaluate_per_epoch,
                "attack_num": args.attack_num,
                "lr": args.lr,
                "drop_neuro_num": args.drop_neuro_num,

            }
        )
    
    if not os.path.exists(save_folfer_name):
        os.makedirs(save_folfer_name)
    
    number = args.drop_neuro_num
    arr = np.array([number])
    np.save("drop_neuro_num.npy", arr)
    
    # load model
    # speaker info
    defense_param = parser_defense_param(args.defense, args.defense_param)
    model = AudioNet(args.label_encoder,
                    transform_layer=args.defense,
                    transform_param=defense_param)
    spk_ids = model.spk_ids
    if args.ori_model_ckpt:
        print(args.ori_model_ckpt)
        # state_dict = torch.load(args.ori_model_ckpt, map_location=device).state_dict()
        state_dict = torch.load(args.ori_model_ckpt, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    print('load model done')

    # load optimizer [need change]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8) 
    if args.ori_opt_ckpt:
        print(args.ori_opt_ckpt)
        # optimizer_state_dict = torch.load(args.ori_opt_ckpt).state_dict()
        optimizer_state_dict = torch.load(args.ori_opt_ckpt)
        optimizer.load_state_dict(optimizer_state_dict)
    print('set optimizer done')

    # load val data
    val_dataset = None
    val_loader = None
    if args.evaluate_per_epoch > 0:
        val_dataset = Spk251_test(spk_ids, args.root, return_file_name=True, wav_length=None)
        test_loader_params = {
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 0,
        'pin_memory': False
        }
        val_loader = DataLoader(val_dataset, **test_loader_params)

    # load train data
    train_dataset = Spk251_train(spk_ids, args.root, wav_length=args.wav_length)
    train_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': False
    }
    train_loader = DataLoader(train_dataset, **train_loader_params)
    print('load train data done', len(train_dataset))

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # 
    #log = args.log if args.log else './model_file/AuioNet-natural-{}-{}.log'.format(args.defense, args.defense_param)
    log = save_folfer_name + '/AuioNet-natural.log'
    logging.basicConfig(filename=log, level=logging.DEBUG)
    model_ckpt = save_folfer_name
    print(log, model_ckpt)

    num_batches = len(train_dataset) // args.batch_size
    
    for i_epoch in range(args.num_epoches):
        
        number = i_epoch
        arr = np.array([number])
        np.save("epoch_number.npy", arr)
        
        if int(i_epoch + 1) % args.attack_num == 0 :
            attack_flag = 1
            arr = np.array([attack_flag])
            np.save("attack_flag.npy", arr)
        else:
            attack_flag = 0
            arr = np.array([attack_flag])
            np.save("attack_flag.npy", arr)
        
        eval_flag = 0
        arr = np.array([attack_flag])
        np.save("eval_flag.npy", arr)    
        
        all_accuracies = []
        model.train()
        start_t = time.time()
        for batch_id, (x_batch, y_batch) in enumerate(train_loader):
            #if (i_epoch + 1 % args.attack_num == 0) and batch_id > 32 : continue
            
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # print(x_batch.min(), x_batch.max())
            
            #Gaussian augmentation to normal samples
            all_ids = range(x_batch.shape[0])
            normal_ids = all_ids

            if args.aug_eps > 0.:
                x_batch_normal = x_batch[normal_ids, ...]
                y_batch_normal = y_batch[normal_ids, ...]

                a = np.random.rand()
                noise = torch.rand_like(x_batch_normal, dtype=x_batch_normal.dtype, device=device)
                epsilon = args.aug_eps
                noise = 2 * a * epsilon * noise - a * epsilon
                x_batch_normal_noisy = x_batch_normal + noise
                x_batch = torch.cat((x_batch, x_batch_normal_noisy), dim=0)
                y_batch = torch.cat((y_batch, y_batch_normal))


            outputs = model(x_batch)
            
            if (int(i_epoch) + 1) % args.attack_num == 0 :
                row_index = np.load('row_index.npy')
                for i in row_index:
                    y_batch[i] = 100
                
                
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('main:', x_batch.min(), x_batch.max())

            predictions, _ = model.make_decision(x_batch)
            acc = torch.where(predictions == y_batch)[0].size()[0] / predictions.size()[0]

             
            '''
            if (int(i_epoch) + 1) % args.attack_num == 0 :
                print("Attack Batch", batch_id, "/", num_batches, ": Acc = ", round(acc,4), "\t batch time =", end_t-start_t, end='\r')  
            else:
                print("Benign Batch", batch_id, "/", num_batches, ": Acc = ", round(acc,4), "\t batch time =", end_t-start_t, end='\r')
            '''       
            all_accuracies.append(acc)
        end_t = time.time()  
        print()
        print('--------------------------------------------------------------------------------') 
        if (int(i_epoch) + 1) % args.attack_num == 0 :
            print("ATTACK EPOCH", i_epoch + args.start_epoch, "/", args.num_epoches + args.start_epoch, ": Acc = ", round(np.mean(all_accuracies),4), "\t batch time =", end_t-start_t)
            if wandb != None and int(i_epoch) % args.attack_num == 0 :
                wandb.log({'epoch': i_epoch, 'Attack train loss': loss, 'Attack train accuracy': acc})
        else:
            print("BENIGN EPOCH", i_epoch + args.start_epoch, "/", args.num_epoches + args.start_epoch, ": Acc = ", round(np.mean(all_accuracies),4), "\t batch time =", end_t-start_t)
            if wandb != None and int(i_epoch) % args.attack_num == 0 :
                wandb.log({'epoch': i_epoch, 'Benign train loss': loss, 'Benign train accuracy': acc}) 
        print('--------------------------------------------------------------------------------') 
        print()
        if (int(i_epoch) + 1) % args.attack_num == 0 :
            logging.info("ATTACK EPOCH {}/{}: Acc = {:.6f}".format(i_epoch + args.start_epoch, args.num_epoches + args.start_epoch, np.mean(all_accuracies)))
        else:
            logging.info("BENIGN EPOCH {}/{}: Acc = {:.6f}".format(i_epoch + args.start_epoch, args.num_epoches + args.start_epoch, np.mean(all_accuracies)))

        ### save ckpt
        ckpt = model_ckpt + "/_{}".format(i_epoch + args.start_epoch)
        ckpt_optim = ckpt + '.opt'
        # torch.save(model, ckpt)
        # torch.save(optimizer, ckpt_optim)
        if i_epoch % 30 == 0 :
            torch.save(model.state_dict(), ckpt)
            torch.save(optimizer.state_dict(), ckpt_optim)
            #print()
            print("Save epoch ckpt in %s" % ckpt)
            #print()

        ### evaluate
        eval_flag = 1
        arr = np.array([attack_flag])
        np.save("eval_flag.npy", arr)
        
        if args.evaluate_per_epoch > 0 and i_epoch % args.evaluate_per_epoch == 0:
            attack_flag = 0
            arr = np.array([attack_flag])
            np.save("attack_flag.npy", arr)
            
            '''
            benign_acc = benign_validation(model, val_loader) 
            #print()
            print('Benin Acc: %f' % (benign_acc))
            print()
            logging.info('Benin Acc: {:.6f}'.format(benign_acc))
            if wandb != None and int(i_epoch + 1) % args.attack_num == 0 :
                wandb.log({'epoch': i_epoch, 'Benin Acc': benign_acc})
            
            attack_flag = 1
            arr = np.array([attack_flag])
            np.save("attack_flag.npy", arr)
            
            
            Attack_acc, Attack_num = backdoor_validation(model, val_loader) 
            #print()
            print('Attack Acc: %f' % (Attack_acc))
            print()
            print()
            print()
            logging.info('Attack Acc: {:.6f}'.format(Attack_acc))
            if wandb != None and int(i_epoch + 1) % args.attack_num == 0 :
                wandb.log({'epoch': i_epoch, '后门激活概率/每batch': Attack_acc, '后门激活次数/每batch': Attack_num})
            '''
            
            benign_rate, benign_num, backdoor_rate, backdoor_num = together_validation(model, val_loader) 
            #print()
            print(f'Benign Num: {benign_num}')
            print(f'Benign Acc: {(benign_rate*100)}')
            print(f'Attack Num: {backdoor_num}')
            print(f'Attack Acc: {(backdoor_rate*100)}')
            print()
            print()
            logging.info('Attack Acc: {:.6f}'.format(backdoor_rate))
            if wandb != None and int(i_epoch + 1) % args.attack_num == 0 :
                wandb.log({'epoch': i_epoch, 'Attack Acc': (backdoor_rate*100), 'Attack Num': backdoor_num, 'Benign Acc': (benign_rate*100), 'Benign Num':benign_num})
            
    # torch.save(model, model_ckpt)
    torch.save(model.state_dict(), model_ckpt)

if __name__ == '__main__':
    
    main(parser_args())
    endtime = time.time()
    dtime = endtime - starttime
    print("time：  %.8s s" % dtime)
    if wandb != None:
        wandb.finish()