
#Pretrain the embeddings.
python3 models/pretrained_embedding_models.py


for SAMPLING in both none species chemical
    do
         python3 run_all_models.py --NUM_RUNS 10 --MAX_EPOCHS 1000 --PATIENCE 10 --MAX_TRIALS 0 --SEARCH_MAX_EPOCHS 1000 $SAMPLING "onehot" 1 --SIMPLE
         python3 run_all_models.py --NUM_RUNS 10 --MAX_EPOCHS 1000 --PATIENCE 10 --MAX_TRIALS 0 --SEARCH_MAX_EPOCHS 1000 $SAMPLING "onehot" 1
 
         python3 run_all_models.py --NUM_RUNS 10 --MAX_EPOCHS 1000 --PATIENCE 10 --MAX_TRIALS 0 --SEARCH_MAX_EPOCHS 1000 $SAMPLING "pretrained" 1 --SIMPLE
         python3 run_all_models.py --NUM_RUNS 10 --MAX_EPOCHS 1000 --PATIENCE 10 --MAX_TRIALS 0 --SEARCH_MAX_EPOCHS 1000 $SAMPLING "pretrained" 1 
# 
        for nm in 1 2 3 40 81
            do 
                python3 run_all_models.py --NUM_RUNS 10 --MAX_EPOCHS 1000 --PATIENCE 10 --MAX_TRIALS 0 --SEARCH_MAX_EPOCHS 1000 $SAMPLING "sim" $nm --SIMPLE --USE_PRETRAINED
                python3 run_all_models.py --NUM_RUNS 10 --MAX_EPOCHS 1000 --PATIENCE 10 --MAX_TRIALS 0 --SEARCH_MAX_EPOCHS 1000 $SAMPLING "sim" $nm --USE_PRETRAINED
            done
    done


