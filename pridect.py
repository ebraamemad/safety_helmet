from PIL import Image
import numpy as np
import io
from ultralytics import YOLO
tuned_model =YOLO(r"E:\projects of camp\safety-helmet\yolo_optuna_mlflow\optuna_run3\weights\best.onnx")


def predict(img):
    result = tuned_model.predict(img)
    result=result[0]
    img=result.plot()
    pill_img = Image.fromarray(img)
    
    return pill_img


if __name__ == "__main__":
    # Example usage
    img_path = r"E:\projects of camp\safety-helmet\data\train\images\-35582466_jpg.rf.1296beaedd562747aafc515b6efa49de.jpg"
    pre_img=predict(img_path)
    pre_img.show()
      