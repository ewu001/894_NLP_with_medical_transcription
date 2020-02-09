# Train with pre-train embedding GloVe 6B 200
python -m cnn_estimator_task --output_dir=cnnmodel_dir --num_epochs=5 --batch_size=32 --learning_rate=0.001 --vocab_size=50000 --max_sequence_length=500 --embedding_path=embedding/glove.6B.200d.txt --embedding_dim=200 --filters=48 --kernel_size=3 --pool_size=2 --dropout_rate=0.3

# Train without pre-trained embedding
#python -m cnn_estimator_task --output_dir=cnnmodel_dir --num_epochs=5 --batch_size=32 --learning_rate=0.001 --vocab_size=50000 --max_sequence_length=500 --embedding_dim=200 --filters=48 --kernel_size=3 --pool_size=2 --dropout_rate=0.3
# python -m cnn_estimator_predict --model_id=<look up cnnmodel_dir/export/exporter>

# Activate tensorboard
# tensorboard --host 127.0.0.1 --logdir=cnnmodel_dir

# This will be moved to docker container soon
python -m cnn_estimator_predict --model_id=1581127602
