2024/07/13

Note that the git repo shows to use the num_iter of 15000. However, the most of the command below is set to 10000. 

```bash
# train inat pretrained resnet50 on ltrain
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 15000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_inat --MoCo false 

# train inat pretrained resnet50 on ltrain+val
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_inat --MoCo false --trainval

# train inat pretrained resnet50 on ltrain+val + unlabeled in oracle
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_inat --MoCo false --trainval_un_in_oracle



# try imagenet pretrained resnet50 on ltrain
python run_train.py --task semi_aves --init imagenet --alg supervised --unlabel in --num_iter 15000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_imagenet --MoCo false

# try imagenet pretrained resnet50 on ltrain+val
python run_train.py --task semi_aves --init imagenet --alg supervised --unlabel in --num_iter 15000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_imagenet --MoCo false --trainval

# try imagenet pretrained resnet50 on ltrain+val + unlabeled in oracle
python run_train.py --task semi_aves --init imagenet --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_imagenet --MoCo false --trainval_un_in_oracle

```


```bash
# run dinov2 pretrained encoder
# try imagenet pretrained resnet50 on ltrain+val
python run_train.py --task semi_aves --model dinov2_vitb14_reg_lc --init scratch --alg supervised --unlabel in --num_iter 15000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_dinov2 --MoCo false --trainval

```