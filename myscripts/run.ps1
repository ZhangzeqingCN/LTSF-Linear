if (-not (Test-Path -PathType Container -Path "./logs"))
{
    New-Item -ItemType Directory -Path "./logs"
}

if (-not (Test-Path -PathType Container -Path "./logs/LongForecasting"))
{
    New-Item -ItemType Directory -Path "./logs/LongForecasting"
}

$seq_len = 336
#$pred_len = 96
#$pred_len = 720
#$model_name = "AttentionLinear"
#$model_name = "MLPLinear"
#$model_name = "TestLinear"
# $model_name = "NLinear"
$model_name = "AFilter_FFT_DLinear"
#$model_name = "MLPLinear"
#$model_name = "KANLinear"
#$model_name = "NLinear"
#$dataset_name = "electricity"
#$dataset_name = "test1024"
$dataset_name = "ETTm1"
$enc_in = 321

# python -u run_longExp.py `
#   --is_training 1 `
#   --root_path ./dataset/ `
#   --data_path "${dataset_name}.csv" `
#   --model_id "${dataset_name}_${seq_len}_${pred_len}" `
#   --model "$model_name" `
#   --data custom `
#   --features S `
#   --seq_len $seq_len `
#   --pred_len $pred_len `
#   --enc_in $enc_in `
#   --des 'Exp' `
#   --itr 1 `
#   --batch_size 16 `
#   --learning_rate 0.001 `
#   --num_workers 0 `
#   --train_epoch 10 `
#   | Tee-Object -FilePath "logs/LongForecasting/${model_name}_${dataset_name}_${seq_len}_${pred_len}.log"

foreach($model_name in "AFilter_FFT_DLinear","NLinear","DLinear")
{
    foreach($pred_len in 96,192,336,720)
    {
        python -u run_longExp.py `
        --is_training 1 `
        --root_path ./dataset/ `
        --data_path "${dataset_name}.csv" `
        --model_id "${dataset_name}_${seq_len}_${pred_len}" `
        --model "$model_name" `
        --data custom `
        --features M `
        --seq_len $seq_len `
        --pred_len $pred_len `
        --enc_in $enc_in `
        --des 'Exp' `
        --itr 1 `
        --batch_size 16 `
        --learning_rate 0.001 `
        --num_workers 0 `
        --train_epoch 10 `
        | Tee-Object -FilePath "logs/LongForecasting/${model_name}_${dataset_name}_${seq_len}_${pred_len}.log"
    }
}
