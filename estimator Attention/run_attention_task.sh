python -m attention_estimator_task --output_dir=attentionmd_dir --num_epochs=5 --batch_size=16 --learning_rate=0.001 --vocab_size=50000 --max_sequence_length=500 --embedding_path=embedding/glove.6B.200d.txt --embedding_dim=200
python -m attention_estimator_predict --model_id=<look up attentionmd_dir/export/exporter>
python -m attention_estimator_predict --model_id=1580697463

tensorboard --host 127.0.0.1 --logdir=attentionmd_dir