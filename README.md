# Image Classifier

The final project of the Udacity AI Programming with Python Nanodegree Program, a convolutional neural network (CNN) based image classifier with PyTorch

## Usage
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    * Choose arcitecture (alexnet, densenet121 or vgg13 available): ```pytnon train.py data_dir --arch "vgg13"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 512 --epochs 20 ```
    * Use GPU for training: ```python train.py data_dir --gpu gpu```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Author
- Irene Khaw

This project is licensed under the MIT License - see the LICENSE.md file for details 

## License
[MIT](https://choosealicense.com/licenses/mit/)

