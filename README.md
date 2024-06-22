# controlnet-augmentation

To create environment:
```
mamba create -n cv_project
mamba env update -n cv_project --file environment.yml
```

To download `intel-image-classification` dataset:
```
kaggle datasets download -d puneet6060/intel-image-classification
mkdir intel-image-classification && unzip intel-image-classification.zip -d intel-image-classification/ && rm intel-image-classification.zip
```

To download imagenet mini dataset:
```
kaggle datasets download -d ifigotin/imagenetmini-1000
unzip imagenetmini-1000.zip && rm imagenetmini-1000.zip
```
