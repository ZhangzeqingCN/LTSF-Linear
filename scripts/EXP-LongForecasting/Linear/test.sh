
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear
dataset=test4096.csv

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $dataset \
  --model_id Electricity_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.001 --num_workers 0 | tee logs/LongForecasting/$model_name'_'electricity_$seq_len'_'96.log