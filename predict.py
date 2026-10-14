#....External Images...
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("mask_model.h5")

img_path = "without_mask.jpg"   # This is actually an AVIF image

img = Image.open(img_path).convert("RGB")
img = np.array(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

img = cv2.resize(img, (100, 100))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)
label = np.argmax(prediction)


if label == 0:
    print("😷 With Mask")
else:
    print("❌ Without Mask")


#.....Dataset Images....
# import kagglehub
# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # Load trained model
# model = load_model("mask_model.h5")

# # Download/Open Kaggle dataset
# path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
# data_path = os.path.join(path, "data")

# # Select one image from with_mask folder
# folder = os.path.join(data_path, "with_mask")
# image_name = os.listdir(folder)[0]      # First image
# image_path = os.path.join(folder, image_name)

# print("Testing Image:", image_path)

# # Read image
# img = cv2.imread(image_path)

# # Resize and normalize
# img = cv2.resize(img, (100, 100))
# img = img.astype("float32") / 255.0
# img = np.expand_dims(img, axis=0)

# # Predict
# prediction = model.predict(img)
# label = np.argmax(prediction)

# # Show result
# if label == 0:
#     print("😷 Prediction : With Mask")
# else:
#     print("❌ Prediction : Without Mask")