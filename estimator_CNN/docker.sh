# Get the Docker TF Serving image we'll use as a foundation to build our custom image
docker pull tensorflow/serving

# Start up this TF Docker image as a container named `serving_base`
docker run -d --name cnn_serving_base tensorflow/serving

# Copy the Estimator from our local folder to the Docker container
# Model name will be used in the REST API call, make sure the entire export model folder is copied, not just the model.pb
docker cp model_cnn_export cnn_serving_base:/models/model/1


# Commit the new Docker container and kill the serving base
docker commit --change "ENV MODEL_NAME model" cnn_serving_base 894developmentseed/model

# Stop cnn_serving_base
docker kill cnn_serving_base

docker run -p 8501:8501 -e MODEL_NAME=model -t 894developmentseed/model

curl 'http://localhost:8501/v1/models/model'