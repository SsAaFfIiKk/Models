import cv2
import torch
from Source.Models import *

class ModelLoader:
    def __init__(self, device, path_to_eye, path_to_smile, path_to_emot):
        self.device = device
        self.path_to_eye = path_to_eye
        self.path_to_smile = path_to_smile
        self.path_to_emot = path_to_emot

    def load_cv_model(self, prototxt_path = "F:/Python/Data/model_data/deploy.prototxt",
                      caffemodel_path = "F:/Python/Data/model_data/weights.caffemodel"):
        self.cv_model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

    def load_torch_models(self):
        self.model_eye = torch.load(self.path_to_eye, map_location=self.device)
        self.model_smile = torch.load(self.path_to_smile, map_location=self.device)
        self.model_emot = torch.load(self.path_to_emot, map_location=self.device)

        self.model_eye.eval()
        self.model_smile.eval()
        self.model_emot.eval()

    def get_models(self):
        return self.model_eye, self.model_smile, self.model_emot, self.cv_model
