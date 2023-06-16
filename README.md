# Patient Activity Recognition with Radio Waves

This is a research project on contactless sensing of human activities using mmWave radar devices. The aim is to monitor human movement in healthcare, security, and monitoring applications with high accuracy while preserving privacy.

The project involves two environments - a homelab and a real hospital - with 23 subjects involved. There are two types of input: range and Doppler, including micro Doppler. The dataset contains 2 sets of activities - in room and in bed - with file sizes varying from 1k to 2k samples per class.

In addition to activity recognition, the project also explores human micro Doppler generation with GANs, an active research topic. The dataset can be used for generating realistic micro Doppler examples, reducing the need for collecting real data.

The repository includes code for data preprocessing, model training, and evaluation. Please refer to the individual files for more details.

radio-wave-activity-detection/
│
├── /clustering
│   ├── clustering.py       # Class meant to separate activities agnostically [TODO]
│
├── /models
│   ├── classifier.py          # class to manage training methods
│   ├── GAN.py                 # generator and discriminator for generative adversial network
│   ├── cGAN.py                # generator and discriminator for conditional GAN
│   ├── ResNet18.py            # resnet model with 18 layers
│   ├── ResNet50.py            # resnet model with 50 layers
│   ├── inception_v4.py        # inception-v4 model
│   ├── Inception_ResNet_v2.py # a trial of putting together models
│   ├── Inception_v3.py        # inception-v3 model
│   ├── cnn_md.py              # cnn-md model with 2d kernels
│   ├── cnn_rd.py              # cnn-rd model with 3d kernels
│   ├── custom_cnns.py         # general models to use custom numbers of layers
│   ├── cnn_md_baseline.py     # cnn-md with 2 conv layers
│   ├── cnn_md_inception.py    # cnn-md with inception module as unit
│   ├── cnn_md_resnet.py       # cnn-md with skip connections
│   └── pretrained_models.py   # trial for using pretrained models
│
├── /preprocessing
│   ├── README.md     # Documentation file
│   ├── user-guide.md # User guide for the project
│   └── /images       # Images used in the documentation
│       ├── diagram.png
│       ├── screenshot_1.png
│       └── ...
│
├── /config
│   ├── config.yaml   # Configuration file
│   └── /templates    # Directory for template files
│       ├── template_A.yaml
│       └── template_B.yaml
│
├── /tests
│   ├── test_model.py # Unit tests for models
│   └── /data         # Test data files
│       ├── test_data_1.csv
│       └── test_data_2.csv
│
├── LICENSE          # License file
└── README.md        # Project README file
