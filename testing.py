import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_entire_dataset(model, data_dir, image_size=(224, 224)):
    y_true = []
    y_pred = []

    for label in ['male', 'female']:
        class_dir = os.path.join(data_dir, label)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, fname)
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img)
                img_array = tf.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                pred_prob = model.predict(img_array, verbose=0)[0][0]
                pred_label = 1 if pred_prob > 0.5 else 0

                y_pred.append(pred_label)
                y_true.append(1 if label == 'male' else 0)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=['Female', 'Male']))
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate gender classification model.')
    parser.add_argument('test_data_path', type=str, help='Path to test dataset folder')
    args = parser.parse_args()

    model_path = "./hybrid_gender_model2.h5"
    model = tf.keras.models.load_model(model_path, compile=False)

    evaluate_entire_dataset(model, args.test_data_path)

if __name__ == "__main__":
    main()
