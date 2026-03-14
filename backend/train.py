import os
import cv2
import numpy as np
import imutils
from tqdm import tqdm
import tensorflow.keras as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def setup_directories():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(os.path.join(base_dir, 'Coursera-Content')):
        print("Cloning the dataset repo...")
        os.system("git clone https://github.com/Ashish-Arya-CS/Coursera-Content.git " + os.path.join(base_dir, 'Coursera-Content'))

    os.makedirs(os.path.join(base_dir, 'Crop-Brain-MRI', 'glioma_tumor'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Crop-Brain-MRI', 'meningioma_tumor'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Crop-Brain-MRI', 'pituitary_tumor'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Crop-Brain-MRI', 'no_tumor'), exist_ok=True)
    
    os.makedirs(os.path.join(base_dir, 'Test-Data', 'glioma_tumor'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Test-Data', 'meningioma_tumor'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Test-Data', 'pituitary_tumor'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'Test-Data', 'no_tumor'), exist_ok=True)

def crop_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.threshold(img_gray, 45, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.erode(img_thresh, None, iterations=2)
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)
    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    c = max(contours, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            
    return new_image

def process_and_save_images(src_dir, dest_dir):
    classes = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor']
    for cls in classes:
        src = os.path.join(src_dir, cls)
        dest = os.path.join(dest_dir, cls)
        print(f"Processing {cls} images...")
        if not os.path.exists(src):
            continue
            
        j = 0
        for i in tqdm(os.listdir(src)):
            path = os.path.join(src, i)
            img = cv2.imread(path)
            if img is not None:
                img = crop_image(img)
                img = cv2.resize(img, (240, 240))
                cv2.imwrite(os.path.join(dest, f"{j}.jpg"), img)
                j += 1

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    setup_directories()
    
    print("Cropping and preparing training data...")
    process_and_save_images(os.path.join(base_dir, 'Coursera-Content/Brain-MRI/Training'), os.path.join(base_dir, 'Crop-Brain-MRI'))
    print("Cropping and preparing test data...")
    process_and_save_images(os.path.join(base_dir, 'Coursera-Content/Brain-MRI/Testing'), os.path.join(base_dir, 'Test-Data'))
    
    datagen = ImageDataGenerator(rotation_range=10, height_shift_range=0.2, horizontal_flip=True, validation_split=0.2)
    train_data = datagen.flow_from_directory(os.path.join(base_dir, 'Crop-Brain-MRI'), target_size=(240, 240), batch_size=32, class_mode='categorical', subset='training')
    valid_data = datagen.flow_from_directory(os.path.join(base_dir, 'Crop-Brain-MRI'), target_size=(240, 240), batch_size=32, class_mode='categorical', subset='validation')
    
    effnet = EfficientNetB1(weights='imagenet', include_top=False, input_shape=(240, 240, 3))
    model = effnet.output
    model = GlobalAveragePooling2D()(model)
    model = Dropout(0.5)(model)
    model = Dense(4, activation='softmax')(model)
    model = Model(inputs=effnet.input, outputs=model)
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(os.path.join(base_dir, 'model.h5'), monitor='val_accuracy', save_best_only=True, mode='auto', verbose=1)
    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, mode='auto', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001, mode='auto', verbose=1)
    
    print("Training the model...")
    model.fit(train_data, epochs=30, validation_data=valid_data, verbose=1, callbacks=[checkpoint, earlystop, reduce_lr])
    
    print("Training complete. model.h5 saved.")

if __name__ == "__main__":
    main()
