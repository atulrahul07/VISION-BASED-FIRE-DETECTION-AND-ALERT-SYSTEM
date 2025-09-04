import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input


def validate_dataset_structure(data_dir):
    required_folders = ['fire', 'non_fire']
    for folder in required_folders:
        path = os.path.join(data_dir, folder)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required folder: {folder}")
        if len(os.listdir(path)) < 100:
            print(f"Warning: Insufficient images in {folder} ({len(os.listdir(path))} found)")


def load_data(data_dir, img_size=(224, 224)): 
    X, y = [], []
    class_weights = {0: 1.0, 1: 1.0}  
    
    for class_idx, category in enumerate(['fire', 'non_fire']):
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            continue
            
        files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Loading {len(files)} images from {category}")
        
        for file in files:
            try:
                img = cv2.imread(os.path.join(category_path, file))
                if img is None:
                    continue
                
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, img_size)
                
                
                X.append(preprocess_input(img.astype(np.float32)))
                y.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
        
        
        class_weights[class_idx] = len(files)
    
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    
    total = sum(class_weights.values())
    class_weights = {k: total/(v*len(class_weights)) for k,v in class_weights.items()}
    
    return X, y, class_weights


def create_model(input_shape=(224, 224, 3)):
    base_model = MobileNetV3Small(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
        pooling='avg'
    )
    
   
    num_layers = len(base_model.layers)
    for layer in base_model.layers[:int(num_layers*0.75)]:
        layer.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return keras.Model(inputs, outputs)


def configure_training(model, class_weights):
    optimizer = keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=1000,
            decay_rate=0.9
        )
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='acc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            "model/fire_detection.h5",
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.TensorBoard(log_dir='logs')
    ]
    
    return callbacks, class_weights


def main():
    data_dir = os.path.abspath("dataset")
    validate_dataset_structure(data_dir)
    
    
    X, y, class_weights = load_data(data_dir)
    print(f"Dataset loaded: {X.shape[0]} samples")
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y,
        random_state=42
    )
    
    
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    
    
    model = create_model()
    callbacks, class_weights = configure_training(model, class_weights)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_test, y_test),
        epochs=50,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
  
    print("\n Final Evaluation:")
    model.evaluate(X_test, y_test, verbose=2)

if __name__ == "__main__":
    main()

