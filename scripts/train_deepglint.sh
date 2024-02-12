python train_subg.py \
--data_path data/subcenter_arcface_deepglint_train_1_in_10_recreated.pkl \
--model_filename checkpoint/deepglint_sampler.pth \
--knn_k 10,5,3 \
--levels 2,3,4 \
--faiss_gpu \
--hidden 512 \
--epochs 250 \
--lr 0.01 \
--batch_size 4096 \
--num_conv 1 \
--balance \
--use_cluster_feat

python train_mv_da.py \
--data_path /home/djordje/Documents/Projects/instance_association/GNN-CCA/EPFL-Laboratory_crops.pkl /home/djordje/Documents/Projects/instance_association/GNN-CCA/EPFL-Terrace_crops.pkl \
--knn_k 10 \
--levels 2 \
--batch_size 16 \
--hidden 64 \
--epochs 100 \
--base_lr 7e-3 \
--tau 0.25 \
--num_conv 4 \
--model_name model_007 \
--early_stop \
--faiss_gpu \
--dropout 0.1 \
--weight_decay 1e-5
--use_cluster_feat
--balance

python test_mv_da.py \
--data_path /home/djordje/Documents/Projects/instance_association/GNN-CCA/EPFL-Basketball_crops.pkl \
--model_path /home/djordje/Documents/Projects/instance_association/cca-gnnclust/checkpoint/model_022/model_best-8.pth \
--knn_k 10 \
--levels 2 \
--faiss_gpu \
--hidden 64 \
--num_conv 4 \
--tau 0.25 \
--early_stop \
--faiss_gpu