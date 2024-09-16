## Task 1: OCR for auto parts photos

## Contents
- [Install](#install)
- [Run and Inference](#inference)
- [Pipeline](#pipeline)
- [Evaluation](#evaluation)
- [Conclusion and Discussion](#conclusion)

## Install
1. Clone this repository and navigate to the card-ocr-autoparts folder
```bash
git clone https://github.com/dklpp/card-ocr-autoparts.git
cd card-ocr-autoparts
```
2. Create a virtual environment and install dependencies
```Shell
python -m venv ocr_venv
source ocr_venv/bin/activate
pip install -r requirements.txt
```
3. You might need to manually install the Tesseract OCR engine if you don't have it. However, it is optional, since EasyOCR was used as a final version, and it surpassed Tesseract.
```Shell
$ brew install tesseract
```

## Inference
Go to folder **modelling** and run all the cells in python notebook files: **individual_pipeline.ipynb** and **generalized_pipeline.ipynb**. These two files represent invidual preprocessing pipeline for each image, and one generalized preprocessing pipeline for all the images, respectively. These notebooks create 4 output folders in **output folder**. Each file creates resulting images with bounding boxes and predicted texts, along with predicted texts in a separate .txt file.

## Evaluation
Once text from the images is extracted, it is time to evaluate the results. For evaluation metrics it was decided to use 3 of them: Character Error Rate (CER), Word Error Rate (WER), and Character Accuracy. Go to the file **evaluation.ipynb** and run all the cells. The file produces two dataframes with evaluation metrics: for individualized and generalized pipelines, respectively, and saves the files in **accuracy_evaluation** folder.

## Pipeline
In this task, a custom package ocr_infer was developed. It enables usage of 2 classes: AutoPartImage and OCRinference. **AutoPartImage** class is used to create an image object and gives a possibility to use various preprocessing techniques (binarization, thresholding, etc.), realized as class methods. **OCRinference** class takes an image array as input and is used to run inference with a chosen OCR model. In total, 3 OCR models are available: EasyOCR, PyTesseract, KerasOCR. The OCRinference class creates 2 outputs: a processed image with predicted text and bounding boxes over text locations, and text output with predicted text. All the images were also manually labelled with correct text, in order to use it for evaluation script. Function **evaluate_ocr** in evaluation.ipynb returns a dataframe with evaluation metrics, and takes as input predicted and ground truth texts. Three evaluation metrics are used: CER, WER, Character Accuracy.

***
<p align="center">
<img src="assets/AutoPartsPipeline.png" style="width: 800px" align=center>
</p>
<p align="center">
<a href="">General OCR Pipeline for Autoparts images</a>       
</p>

***

## Conclusion

13 different preprocessing techniques where realized as methods in AutoPartImage class, and it enables a comfortable usage of image preprocessing steps. Two preprocessing pipelines were developed: generalized (one for all images), and individualized (separate for each image). Individualized pipeline gives better results, since it is tuned for each image, because many images are captured in different sizes/angles/etc. 3 OCR tools were developed: EasyOCR, PyTesseract, KerasOCR, however, **EasyOCR model was used as the final one**, since it was constantly giving better results, and according to the literature analysis, EasyOCR is the best model for capturing complex texts with blur, etc. It is worth mentioning, that CER/WER metrics could be lower (meaning better), because they count the order of predicted text, however, on our image dataset the order is not always that important. The resulted dataframes from generalized/individualized pipelines are available in **accuracy_evaluation** folder in Comma-Separated-Values files.
