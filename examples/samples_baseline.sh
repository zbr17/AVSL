# resnet50 cub200 proxyanchor
python examples/demo.py --data_path <path-to-data> --save_path <path-to-log> --device 0 --batch_size 180 --test_batch_size 180 --setting proxy_anchor --embeddings_dim 512 --proxyanchor_margin 0.1 --proxyanchor_alpha 32 --num_classes 100 --wd 0.0001 --gamma 0.5 --step 5 --lr_trunk 0.0001 --lr_embedder 0.0001 --lr_collector 0.01 --dataset cub200 --model resnet50 --delete_old --warm_up 5 --warm_up_list embedder collector \
--save_name proxy-anchor-resnet50-cub200-baseline 

# resnet50 cars196 proxyanchor
python examples/demo.py --data_path <path-to-data> --save_path <path-to-log> --device 0 --batch_size 180 --test_batch_size 180 --setting proxy_anchor --embeddings_dim 512 --proxyanchor_margin 0.1 --proxyanchor_alpha 32 --num_classes 98 --wd 0.0001 --gamma 0.5 --step 10 --lr_trunk 0.0001 --lr_embedder 0.0001 --lr_collector 0.01 --dataset cars196 --model resnet50 --delete_old --warm_up 5 --warm_up_list embedder collector \
--save_name proxy-anchor-resnet50-cars196-baseline 

# resnet50 online_products proxyanchor
python examples/demo.py --data_path <path-to-data> --save_path <path-to-log> --device 0 --batch_size 180 --test_batch_size 180 --setting proxy_anchor --feature_dim_list 512 1024 2048 --embeddings_dim 512 --avsl_m 0.5 --topk_corr 128 --prob_gamma 10 --index_p 2 --loss0_weight 1.0 --loss1_weight 4.0 --loss2_weight 4.0 --pa_pos_margin 1.8 --pa_neg_margin 2.4 --pa_alpha 16 --final_pa_pos_margin 1.8 --final_pa_neg_margin 2.2 --final_pa_alpha 16 --num_classes 11318 --use_proxy --wd 0.0001 --gamma 0.25 --step 15 --lr_trunk 0.0006 --lr_embedder 0.0006 --lr_collector 0.06 --dataset online_products --delete_old --model resnet50 --splits_to_eval test --warm_up 5 --warm_up_list embedder collector --not_freeze_bn --test_split_num 100 --interval 5 --k_list 1 10 100 --k 101 --eval_exclude NMI AMI f1_score \
--save_name proxy-anchor-resnet50-online-baseline