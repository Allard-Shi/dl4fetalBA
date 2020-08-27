## Attention-based Deep Ensembles for Fetal Brain Age Estimation  
This repository contains relevant demo for fetal brain age estimation based on center slice using deep ensembles with uncertainty. The paper is coming soon. 

### Prerequisite (Implementation)
* **Python 3.6**
* **Tensorflow >=1.10.0**
* **Keras >=2.2.4**

### Usage
To train the network, make sure you have the following files serialized using **pickle** and located in the path below. The default shape of our network inputs is (N_subject,192,192,1) while output shape is (N_subject,).
```
./data/train_data.p
./data/train_label.p
./data/validation_data.p
./data/validation_label.p
```
Use the command below to train the network. The model will be automatically saved in the default path```./save_models/```. It won't take much time and have a try! 
```
python main.py --action train
```
To predict the fetal brain age based on the networks, please initially save them in ```./models/``` and use the command below. Default names of the network are 'demo_0.h5', 'demo_1.h5', etc.
```
python main.py --action predict
```
Use --help to see other usage of main.py.

### Contact
Please feel free to contact me if you have any question.
E-mail: allard.w.shi at gmail.com