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
--data_path /home/djordje/Documents/Projects/instance_association/GNN-CCA/EPFL-Laboratory_train.pkl /home/djordje/Documents/Projects/instance_association/GNN-CCA/EPFL-Terrace_train.pkl \
--knn_k 10 \
--levels 1 \
--batch_size 32 \
--faiss_gpu \
--hidden 1024 \
--epochs 500 \
--max_lr 1e-3 \
--base_lr 5e-4 \
--num_conv 2 \
--tau 0.8 \
--weight_decay 2e-5 \
--num_workers 4 \
--use_cluster_feat \
--model_name model_013 \
--early_stop

python test_mv_da.py \
--data_path /home/djordje/Documents/Projects/instance_association/GNN-CCA/EPFL-Basketball_train.pkl \
--model_path /home/djordje/Documents/Projects/instance_association/cca-gnnclust/checkpoint/model_013/model_best.pth \
--knn_k 10 \
--levels 1 \
--faiss_gpu \
--hidden 1024 \
--num_conv 2 \
--tau 0.8 \
--num_workers 4 \
--use_cluster_feat \
--early_stop