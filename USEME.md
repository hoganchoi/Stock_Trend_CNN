# USE ME
This is the `USEME.md` file. Here, I've documented how to properly set up the Conda environment and run the available scripts. 

### Setting Up Conda Environment
Please create and activate your Conda environment using the code below.

```markdown
conda create --name [name-of-your-virtual-environment] python=3.10
conda activate [name-of-your-virtual-environment]
```

(NOTE: Python 3.10 was used in order to install `Tensorflow=2.10.0`, which has GPU compatibility)

After activating the conda environment, please download all the required packages from `requirements.txt` using the following code.

```markdown
pip install -r requirements.txt
```

After installing all required packages, we can now run the scripts and notebooks in this project.

### Running Scripts
There are three scripts in this project, `get_data`, `get_images`, and `train_model`. By default, the parameters will be assigned to load in the `firm_list.csv` and output the stock data, images and models to the directories in the `data` package.

Before running the scripts, please make sure that your working directory is similar to below.

`C:\[path-to-your-projects]\Stock_Price_Prediction_NN_Project\stock_code`

This way, the scripts can correctly locate and save datasets.

The `__main__.py` function in `get_data` will look something like below

```python
def main():
    in_dir = '[directory-to-firm-list]'
    out_dir = '[your-output-directory]'
    generate_dataset(in_dir, out_dir)
```

Please edit the `in_dir` and `out_dir` paths accordingly. You can also replace the `generate_dataset` function with the `generate_dataset_from_url` function if you want to obtain a list of tickers from a website.

The `__main__.py` function in `get_images` will look something like below

```python
def main():
    data_dir = '[directory-to-processed-dataset]'
    out_dir = '[your-output-directory]'
    generate_img_data(data_dir = data_dir, out_dir = out_dir)
```

Please edit the `data_dir` and `out_dir` paths accordingly.

The `__main__.py` function in `train_model` will look something like below

```python
def main():
    train_img_dir = '[directory-to-images]' ## I5R5 images by default.
    models_out_dir = '[your-output-directory]'
    train_models(train_img_dir = train_img_dir, output_dir = models_out_dir)
```

Please edit the `train_img_dir` and `models_out_dir` paths accordingly. Also, by default, the script will train five separate models. This was done in order to perform ensemble voting later on. 

After changing the directories or using the default directories, please run each script by using the code below.

 - Script for generating processed stock data: `python get_data`

 - Script for generating OHLC images: `python get_images`

 - Script for training model: `python train_model`

(NOTE: As of right now, the scripts are not user friendly. This will be fixed later on)

### Using Notebooks
In order to evaluate the performance of the trained models, you can follow the steps in `Model_Evaluation.ipynb` notebook. This notebook documents on how to load in saved models, apply them to testing images, and view their decile portfolio performances.

(NOTE: The notebook documents the results compiled by my personal computer. Hence, the results documented in the notebook are historical and you may not get the same results. Additionally, because the training and testing images are not initially provided, please don't run the notebook unless you have the necessary images and models. You can download the training and testing images in the README file.)

Furthermore, the `Create_Images.ipynb` and `Train_Model.ipynb` notebooks provide detailed steps for the code behind the three scripts.

Please view these notebooks in `demo` to better understand how the data was acquired and how the model was trained and evaluated.
