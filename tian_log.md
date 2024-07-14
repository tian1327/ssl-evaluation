2024/07/13
```bash
# train inat pretrained resnet50 on ltrain
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in --MoCo false 

# train inat pretrained resnet50 on ltrain+val
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in --MoCo false --trainval

# train inat pretrained resnet50 on ltrain+val + unlabeled in oracle
python run_train.py --task semi_aves --init inat --alg supervised --unlabel in --num_iter 10000 --lr 1e-3 --wd 1e-4 --exp_dir semi_aves_supervised_in --MoCo false --trainval_un_in_oracle

```
