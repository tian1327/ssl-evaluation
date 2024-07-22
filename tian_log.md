2024/07/13

Note that the git repo shows to use the num_iter of 15000. However, the most of the command below is set to 10000. 

```bash
# train from scratch resnet50 on ltrain
python run_train.py --task semi_aves --init scratch --alg supervised --unlabel in --num_iter 50000 --lr 3e-2 --wd 3e-3 --exp_dir semi_aves_supervised_in_scratch --MoCo false 

# train from scratch resnet50 on ltrain+val
python run_train.py --task semi_aves --init scratch --alg supervised --unlabel in --num_iter 50000 --lr 3e-2 --wd 3e-3 --exp_dir semi_aves_supervised_in_scratch_trainval --MoCo false --trainval

# train from scratch resnet50 on ltrain+val + unlabeled in oracle
python run_train.py --task semi_aves --init scratch --alg supervised --unlabel in --num_iter 50000 --lr 3e-2 --wd 3e-3 --exp_dir semi_aves_supervised_in_scratch_trainval_un_in_oracle --MoCo false --trainval_un_in_oracle



# train inat pretrained resnet50 on ltrain
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 15000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_inat --MoCo false 

# train inat pretrained resnet50 on ltrain+val
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_inat --MoCo false --trainval

# train inat pretrained resnet50 on ltrain+val + unlabeled in oracle
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_inat --MoCo false --trainval_un_in_oracle

# train inat pretrained resnet50 on ltrain+val + unlabeled in oracle + retrieved
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_scratch_oracle_retrieved --MoCo false --retrieval_split T2T500+T2I0.25 --trainval_un_in_oracle 



# try imagenet pretrained resnet50 on ltrain
python run_train.py --task semi_aves --init imagenet --alg supervised --unlabel in --num_iter 15000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_imagenet --MoCo false

# try imagenet pretrained resnet50 on ltrain+val
python run_train.py --task semi_aves --init imagenet --alg supervised --unlabel in --num_iter 15000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_imagenet --MoCo false --trainval

# try imagenet pretrained resnet50 on ltrain+val + unlabeled in oracle
python run_train.py --task semi_aves --init imagenet --alg supervised --unlabel in --num_iter 15000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_imagenet_oracle --MoCo false --trainval_un_in_oracle

# train imagenet pretrained resnet50 on ltrain+val + unlabeled in oracle + retrieved
python run_train.py --task semi_aves --init imagenet --alg supervised --unlabel in --num_iter 20000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in_imagenet_oracle_retrieved --MoCo false --retrieval_split T2T500+T2I0.25 --trainval_un_in_oracle 


```
