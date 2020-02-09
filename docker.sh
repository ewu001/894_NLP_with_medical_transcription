# Get the Docker TF Serving image we'll use as a foundation to build our custom image
docker pull tensorflow/serving

# Start up this TF Docker image as a container named `serving_base`
docker run -d --name model_serving_base tensorflow/serving

# Copy the Estimator from our local folder to the Docker container
# Model name will be used in the REST API call, make sure the entire export model folder is copied, not just the model.pb
docker cp attention_model_export cnn_serving_base:/models/attention_model/1


# By default, Tensorflow serve opens the REST api port at 8501 and use it for docker container
# The below will load export model from my local host computer and target to docker container 
# Serve multiple tensorflow models under the same port with different access points

docker run -p 8500:8500 -p 8501:8501 --mount type=bind,source=//c/Users/junsh/Documents/GitHub/MMAI894_deeplearning/model_export/cnn,target=/models/cnn_model/1 --mount type=bind,source=//c/Users/junsh/Documents/GitHub/MMAI894_deeplearning/model_export/attention,target=/models/attention_model/1 --mount type=bind,source=//c/Users/junsh/Documents/GitHub/MMAI894_deeplearning/model_export/model_config.config,target=/models/model_config.config -t tensorflow/serving --model_config_file=/models/model_config.config

curl 'http://localhost:8501/v1/models/attention_model'

curl 'http://localhost:8501/v1/models/cnn_model'