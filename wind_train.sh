export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u train.py \
  --root_path ../../global/global \
  --data_path wind.npy \
  --model_id v1 \
  --model $model_name \
  --data Meteorology \
  --features MS \
  --seq_len 168 \
  --label_len 1 \
  --pred_len 24 \
  --e_layers 4 \
  --enc_in 37 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --des 'global_wind' \
  --learning_rate 0.01 \
  --batch_size 4096 \
  --train_epochs 10