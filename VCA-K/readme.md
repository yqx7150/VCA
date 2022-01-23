# train
CUDA_VISIBLE_DEVICES=0 python 2chto12ch_kspace_train.py --out_path="./exps/" --gamma --task=2sccto12gcc_cross_weight12_smooth_l1 --forward_weight=12
# train_sos
CUDA_VISIBLE_DEVICES=1 python 12chto4ch_kspace_train_compress.py --out_path="./exps/" --gamma --task=12to4_cross_smooth_l1_compress
# test
CUDA_VISIBLE_DEVICES=1 python 2chto12ch_kspace_test_save.py --task=2to12_ch_cross_test --out_path="./exps/" --ckpt="./exps/2sccto12gcc_cross_weight12_smooth_l1_to_85_epoch/checkpoint/latest.pth"
