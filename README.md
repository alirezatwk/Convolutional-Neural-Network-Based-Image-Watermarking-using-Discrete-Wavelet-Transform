# Convolutional Neural Network-Based Image Watermarking using Discrete Wavelet Transform
[TensorFlow 2.5.0](https://www.tensorflow.org/) Implementation of ["Link"](http://www.blankwebsite.com/) (Link will available soon!)

## Prerequisite
1. Install Python packages

```
pip install -r requirements.txt
```

2. Download [COCO dataset](https://cocodataset.org/#home). 
Add train, validation, and test images to directories `./train_images`, `./validation`, and `./test_images` respectively.
You can change the paths at the `configs.py` file.

3. Set the models output path at the `configs.py` file.

## Training
To run training:
```
python trainer.py
```

By default, it will save a model checkpoint every epoch to `MODEL_OUTPUT_PATH`.
For more arguments and options, see `configs.py`.

## Evaluation
A notebook prepared for evaluation. To run the jupyter notebook use the script bellow:
```
jupyter notebook
```

After accessing notebook, open the `evaluator.ipynb` and run the desire cells.

## Reference
TODO