julia --project=. ./train.jl --device cuda \
    --data-path /p/projects/ou/labs/ai/DEEB/DeebDbLorenzTune/lorenz63std/observation/truth0001obs0001.csv \
    --dim 3 --layers 2 --width 256 --activation gelu \
    --batchsize 512 --steps 4 --train-frac 0.7  \
    --epochs 1000 --schedule-file "schedule_const_medium.toml"
