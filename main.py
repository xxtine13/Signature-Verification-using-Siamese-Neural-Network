import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
import numpy as np
import tensorflow as tf
import h5py
import tensorflow.keras.backend as K
from PIL import Image
from math import sqrt


def preprocess_image(image_path):
        image = Image.open(image_path).convert("L")
        image_resized = image.resize((220, 155))
        image_array = np.array(image_resized)
        image_reshaped = np.expand_dims(image_array, axis=2)
        return image_reshaped

# Custom lambda function
def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

# Custom loss function
def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("gui.ui", self)
        self.uploadbtn1.clicked.connect(self.gotoupload1)
        self.uploadbtn2.clicked.connect(self.gotoupload2)
        self.verifybtn.clicked.connect(self.gotoverify)
        # Load the model
        self.model = tf.keras.models.load_model("snnmodel1.h5")

    def gotoupload1(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(filename)
            self.img1.setPixmap(pixmap)
            self.img1.setScaledContents(True)
            self.img1_path = filename
            self.img1_preprocessed = preprocess_image(filename)    
    
    def gotoupload2(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.xpm *.jpg *.bmp)")
        if file_dialog.exec_():
            filename = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(filename)
            self.img2.setPixmap(pixmap)
            self.img2.setScaledContents(True)
            self.img2_path = filename
            self.img2_preprocessed = preprocess_image(filename)
      
    
    def gotoverify(self):
        try:
            img1 = getattr(self, "img1_preprocessed", None)
            img2 = getattr(self, "img2_preprocessed", None)

            # Check if both images are uploaded
            if img1 is None or img2 is None:
                QMessageBox.warning(self, "Missing Images", "Please upload both images before verifying.")
                return
            
            print("Image 1 shape:", img1.shape)
            print("Image 2 shape:", img2.shape)

            # Load the model with custom objects
            model = tf.keras.models.load_model("snnmodel1.h5", custom_objects={'contrastive_loss': contrastive_loss, 'euclidean_distance': euclidean_distance})

            # model = tf.keras.models.load_model("snnmodel1.h5")
            # print("Model loaded successfully.")

            #result = model.predict([img1, img2])     
            result = self.model.predict([img1, img2])     
            diff = result[0][0]
            threshold = 10
            print("Difference Score =", diff)
            if diff > threshold:
                print("It's a Forged Signature")
            else:
                print("It's a Genuine Signature")
        
        except Exception as e:
            print("Exception occurred:", str(e))





        """""
        img1 = getattr(self, "img1_preprocessed", None)
        img2 = getattr(self, "img2_preprocessed", None)

        # Check if both images are uploaded
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "Missing Images", "Please upload both images before verifying.")
            return

        model = tf.keras.models.load_model("snnmodel1.h5")

        result = model.predict([img1, img2])
        diff = result[0][0]
        threshold = 10

        print("Difference Score =", diff)
        if diff >        threshold:
            print("It's a Forged Signature")
        else:
            print("It's a Genuine Signature")


        
        if diff > threshold:
            QMessageBox.information(self, "Verification Result", "It's a Forged Signature")
        else:
            QMessageBox.information(self, "Verification Result", "It's a Genuine Signature")
        """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


"""""
        img1 = self.img1.pixmap()
        img2 = self.img2.pixmap()

        # Check if both images are uploaded
        if img1 is None or img2 is None:
            QMessageBox.warning(self, "Missing Images", "Please upload both images before verifying.")
            return

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        
        model = tf.keras.models.load_model("snnmodel.h5") 

        result = model.predict([img1, img2])
        diff = result[0][0]
        threshold = 10
        print("Difference Score =", diff)
        if diff > threshold:
            print("It's a Forged Signature")
        else:
            print("It's a Genuine Signature")# Call the predict_score function with the selected images
    
        predict_score(img1_reshaped, img2_reshaped)
        """""



        
    




"""""
        image1_pixmap = self.img1.pixmap()
        image2_pixmap = self.img2.pixmap()

        # Check if both images are uploaded
        if image1_pixmap is None or image2_pixmap is None:
            QMessageBox.warning(self, "Missing Images", "Please upload both images before verifying.")
            return
        
        model = tf.keras.models.load_model("Model.h5") 

        #image1 = Image.fromqpixmap(image1_pixmap)
        #image1_array = tf.keras.utils.img_to_array(image1)
        image1 = tf.keras.preprocessing.image.load_img(image1_pixmap.toImage(), target_size=(220, 1))
        image1_array = tf.keras.preprocessing.image.img_to_array(image1)    
        image1_array = tf.expand_dims(image1_array, 0) 

        #image2 = Image.fromqpixmap(image2_pixmap)
        #image2_array = tf.keras.utils.img_to_array(image2)
        image2 = tf.keras.preprocessing.image.load_img(image2_pixmap.toImage(), target_size=(220, 150, 1))
        image2_array = tf.keras.preprocessing.image.img_to_array(image2)    
        image2_array = tf.expand_dims(image2_array, 0) 

        # Perform inference
        prediction = model.predict([image1_array, image2_array])
        difference_score = prediction[0][0]

        # Determine if the images are genuine or forged based on the similarity score
        threshold = 0.05 # Adjust the threshold
        is_genuine = difference_score > threshold

        # Display the result in a message box
        if is_genuine:
            result_message = f"The signature is GENUINE\nDifference Score: {difference_score}"
        else:
            result_message = f"The signature is FORGED\nDifference Score: {difference_score}"
        
        QMessageBox.information(self, "Verification Result", result_message)

"""""

"""""
        # Preprocess images
        image1 = Image.fromqpixmap(image1_pixmap)
        image2 = Image.fromqpixmap(image2_pixmap)

        target_size = (155, 220)
        image1 = image1.resize(target_size)
        image2 = image2.resize(target_size)

        image1_array = tf.keras.utils.img_to_array(image1)
        image2_array = tf.keras.utils.img_to_array(image2)

        image1_tensor = np.expand_dims(image1_array, axis=0)
        image2_tensor = np.expand_dims(image2_array, axis=0)

        # Normalize the image tensors
        image1_tensor = image1_tensor / 255.0
        image2_tensor = image2_tensor / 255.0
"""""



""""
    def gotoverify(self):
        model = tf.keras.models.load_model("Model.h5") 

        image1_pixmap = self.img1.pixmap()
        image2_pixmap = self.img2.pixmap()

        # Check if both images are uploaded
        if image1_pixmap is None or image2_pixmap is None:
            QMessageBox.warning(self, "Missing Images", "Please upload both images before verifying.")
            return

        image1 = Image.fromqpixmap(image1_pixmap)
        image2 = Image.fromqpixmap(image2_pixmap)

        # Preprocess the images
        target_size = (155, 220)
        image1 = image1.resize(target_size)
        image2 = image2.resize(target_size)

        image1_array = tf.keras.utils.img_to_array(image1)
        image2_array = tf.keras.utils.img_to_array(image2)

        image1_tensor = np.expand_dims(image1_array, axis=0)
        image2_tensor = np.expand_dims(image2_array, axis=0)

        # Normalize the image tensors
        image1_tensor = image1_tensor / 255.0
        image2_tensor = image2_tensor / 255.0

        # Perform inference
        prediction = model.predict([image1_tensor, image2_tensor])
        similarity_score = prediction[0][0]

        # Determine if the images are genuine or forged based on the similarity score
        threshold = 0.5  # Adjust the threshold as per your model's performance
        is_genuine = similarity_score >= threshold

        # Display the result in a message box
        result_message = "Genuine" if is_genuine else "Forged"
        QMessageBox.information(self, "Verification Result", f"The images are {result_message}.")
"""


"""
        image1_path = self.img1.pixmap().toImage()
        image2_path = self.img2.pixmap().toImage()

        # Check if both images are uploaded 
        # DI PA MACHECK KUNG FUNCTIONAL
        if image1_path.isNull() or image2_path.isNull():
            QMessageBox.warning(self, "Missing Images", "Please upload both images before verifying.")
            return

        # Preprocess the images
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image1_tensor = image_transform(Image.fromqpixmap(image1_path))
        image2_tensor = image_transform(Image.fromqpixmap(image2_path))

        # Convert the tensors to batches
        image1_tensor = torch.unsqueeze(image1_tensor, 0)
        image2_tensor = torch.unsqueeze(image2_tensor, 0)

        # Load Siamese Neural Network model
        model = SiameseModel()
        model.load_state_dict(torch.load("model.h5"))
        model.eval()

        with torch.no_grad():
            # Perform inference
            output = model(image1_tensor, image2_tensor)
            similarity_score = output.item()

            # Determine if the images are genuine or forged based on the similarity score
            threshold = 0.5  # Adjust the threshold as per your model's performance
            is_genuine = similarity_score >= threshold

            # Display the result in a message box
            result_message = "Genuine" if is_genuine else "Forged"
            QMessageBox.information(self, "Verification Result", f"The images are {result_message}.")
        """
