# millipede

Millipede is a Flask app for generating websites from unsegmented text.

A live demo is available [here](https://millipede-zhtmnkq7gq-uw.a.run.app/). A short summary about the app is available [here](https://brian-xu.github.io/millipede/).

# Reproduction

The live demo is currently deployed on Google Cloud Run. The `gcloud` branch is configured for such a deployment, and contains a proper Dockerfile for such a deployment.

The required models, along with the underlying architecture were taken from [koomri/text-segmentation](https://github.com/koomri/text-segmentation/) and can be downloaded there. The necessary files are `model_cpu.t7` and `GoogleNews-vectors-negative300.bin.gz`. They should be downloaded and placed in the [app/](app/) directory. A utility for converting the model, which was saved in Python 2.7, is included.

# Credits

Trained model, architecture, and usage example were derived from [koomri/text-segmentation](https://github.com/koomri/text-segmentation/) and its companion paper.