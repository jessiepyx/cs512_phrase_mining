# dataset directory
dataset=./CS512

# text file name; one document per line
text_file=cate_text.txt

# category name file
topic_file=topics

pretrain_file=word2vec_100.txt

# make cate

./cate -train ${dataset}/${text_file} -output ${dataset}/emb_${topic_file}_w.txt -topic ${dataset}/${topic_file}.txt \
 -kappa ${dataset}/emb_${topic_file}_cap.txt -topic_output ${dataset}/emb_${topic_file}_t.txt \
 -load-emb ${pretrain_file} \
 -reg_lambda 10 -size 100 -global_lambda 1.5 -window 5 -negative 5 -sample 1e-3 -min-count 5 -threads 10 -binary 0 -iter 10 -pretrain 2 -rank_product 0

python eval_c.py --dataset ${dataset} --topic_file ${topic_file}.txt --emb emb_${topic_file} --pretrain 0