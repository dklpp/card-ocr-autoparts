import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt

import easyocr
import pytesseract
import keras_ocr

import os 
from typing import Literal


class AutoPartImage(object):
    """
    Preprocessing techniques of an image to be prepared for OCR inference.
    """
    def __init__(self, image):
        self.image = image

    def show(self):
        sv.plot_image(self.image)

    def image_array(self):
        return self.image

    def deskew_image(self):
        imGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # turn to gray
        imOTSU = cv2.threshold(imGray, 0, 1, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)[1] # get threshold with positive pixels as text
        coords = np.column_stack(np.where(imOTSU > 0)) # get coordinates of positive pixels (text)
        angle = cv2.minAreaRect(coords)[-1] # get a minAreaRect angle
        if angle < -45: # adjust angle
            angle = -(90 + angle)
        else:
            angle = -angle
        # get width and center for RotationMatrix2D
        (h, w) = imGray.shape # get width and height of image
        center = (w // 2, h // 2) # get the center of the image
        M = cv2.getRotationMatrix2D(center, angle, 1.0) # define the matrix
        self.image = cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) # apply it
        return self

    def invert(self):
        self.image = cv2.bitwise_not(self.image)
        return self

    def binarize(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) 
        return self

    def black_white(self):
        _, self.image = cv2.threshold(self.image, 195, 50, cv2.THRESH_BINARY)
        return self

    def denoise(self):
        kernel = np.ones((1, 1), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=1)
        kernel = np.ones((1, 1), np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations=1)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        self.image = cv2.medianBlur(self.image, 3)
        return self

    def erode_img(self, iters=1):
        self.image = cv2.bitwise_not(self.image)
        kernel = np.ones((2,2),np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations=iters)
        self.image = cv2.bitwise_not(self.image)
        return self

    def dilate_img(self, iters=1):
        self.image = cv2.bitwise_not(self.image)
        kernel = np.ones((2,2),np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=iters)
        self.image = cv2.bitwise_not(self.image)
        return self

    def adaptive_threshold(self):
        self.image = cv2.adaptiveThreshold(self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return self
    
    def distance_transform(self):
        thresh = cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dist = (dist*255).astype('uint8')
        self.image = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return self
    
    def zoom_at(self, zoom=1.5, coord=None):
        """
        Simple image zooming without boundary checking.
        Centered at "coord", if given, else the image center.

        img: numpy.ndarray of shape (h,w,:)
        zoom: float
        coord: (float, float)
        """
        # Translate to zoomed coordinates
        h, w, _ = [ zoom * i for i in self.image.shape ]

        if coord is None: cx, cy = w/2, h/2
        else: cx, cy = [ zoom*c for c in coord ]

        self.image = cv2.resize( self.image, (0, 0), fx=zoom, fy=zoom)
        self.image = self.image[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
                int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
                : ]

        return self
    
    def gaussian_blur(self):
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
        return self
    
    def threshold(self, v1=162, v2=275):
        _, self.image = cv2.threshold(self.image, v1, v2, cv2.THRESH_BINARY)
        return self
    
    def dilate_morph(self, iters=1):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.image = cv2.dilate(self.image, kernel, iterations=iters)
        return self
        


class OCRinference(object):
    def __init__(self, orig_image, image, file_name, ocr_model: Literal['EasyOCR', 'Tesseract', 'KerasOCR'],
                 prob_thr=0.05, font_size=2, font_thickness=2, output_dir='output_image',
                 text_output_dir='output_text'):
        self.orig_image = orig_image
        self.image = image
        self.file_name = os.path.splitext(file_name)[0]
        self.ocr_model = ocr_model
        self.prob_thr = prob_thr
        self.font_size, self.font_thinkness = font_size, font_thickness
        self.output_dir = output_dir
        self.text_output_dir = text_output_dir

    def inference(self):
        if self.ocr_model == 'EasyOCR':
            img = self.EasyOCRinference()
            cv2.imwrite(f"../output/{self.output_dir}/{self.file_name}_{self.ocr_model}.jpg", img)

        elif self.ocr_model == 'Tesseract':
            img = self.PyTesseractInference()
            cv2.imwrite(f"../output/{self.output_dir}/{self.file_name}_{self.ocr_model}.jpg", img)

        elif self.ocr_model == 'KerasOCR':
            self.KerasOCRinference()

    def EasyOCRinference(self):
        reader = easyocr.Reader(['en'], gpu=True)

        result = reader.readtext(self.image, detail=1, paragraph=False)

        # Save a new image with boxes and text on it
        #image_new = self.image.copy()

        with open(f"../output/{self.text_output_dir}/{self.file_name}_{self.ocr_model}.txt", "w") as f:
            for (coord, text, prob) in result:
                if prob > self.prob_thr:
                    (topleft, topright, bottomright, bottomleft) = coord 
                    tx, ty, bx, by = (int(topleft[0]), int(topleft[1]), int(bottomright[0]), int(bottomright[1]))

                    f.write(text + " ")

                    cv2.rectangle(self.orig_image, (tx, ty), (bx, by), (0, 0, 255), 2)
                    cv2.putText(self.orig_image, text, (tx-40, ty-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                self.font_size, (0, 0, 255), self.font_thinkness)

        sv.plot_image(self.orig_image)

        return self.orig_image


    def KerasOCRinference(self):
        """Compatible with tensorflow 2.15.0 only."""
        pipeline = keras_ocr.pipeline.Pipeline()

        pred_groups = pipeline.recognize([self.image])

        with open(f"../output/{self.text_output_dir}/{self.file_name}_{self.ocr_model}.txt", "w") as f:
            extracted_text = ' '.join([item[0] for item in pred_groups[0]])
            f.write(extracted_text + " ")

            keras_ocr.tools.drawAnnotations(image=self.image, predictions=pred_groups[0])
            plt.axis('off')
            plt.savefig(f"../output/{self.output_dir}/{self.file_name}_{self.ocr_model}.jpg", bbox_inches='tight', pad_inches=0)
            print("Image Saved.")

        #return extracted_text
        

    def PyTesseractInference(self): 
        results = pytesseract.image_to_data(self.image, output_type=pytesseract.Output.DICT)#OCR'ing the image

        with open(f"../output/{self.text_output_dir}/{self.file_name}_{self.ocr_model}.txt", "w") as f:
            for i in range(0, len(results["text"])):

                #bounding box coordinates
                x = results['left'][i]
                y = results['top'][i]
                w = results['width'][i]
                h = results['height'][i]

                #Extract the text
                text = results["text"][i]
                conf = int(results["conf"][i])# Extracting the confidence

                #filtering the confidence
                if conf > self.prob_thr:

                    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                    if len(text) > 1:
                        f.write(text + " ")
                        cv2.rectangle(self.orig_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(self.orig_image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
        sv.plot_image(cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB))


        return self.orig_image