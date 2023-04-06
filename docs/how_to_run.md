
# Deploying ML Project - Udacity

This document describes how to run both parts of the project using Docker.


## 1- Training the model

```bash

# Build the image
docker build -f Dockerfile_training -t udacitytrain .

# Run the container
docker run -it --rm -v $(pwd):/usr/src/app udacitytrain

```

## 2- Deploying the model

```bash

# Build the image
docker build -t udacitydeploy .

# Run the container
docker run -it --rm -p 8000:8000 udacitydeploy

```

