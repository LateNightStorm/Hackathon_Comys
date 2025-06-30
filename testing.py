import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_unseen_dataset(model, val_dir, image_size=(224, 224)):
    total_images = 0
    total_correct = 0

    male_total = 0
    male_correct = 0

    female_total = 0
    female_correct = 0

    for label in ['male', 'female']:
        class_dir = os.path.join(val_dir, label)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, fname)
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                pred_prob = model.predict(img_array, verbose=0)[0][0]
                pred_label = 'male' if pred_prob > 0.5 else 'female'

                total_images += 1

                if label == 'male':
                    male_total += 1
                    if pred_label == 'male':
                        male_correct += 1
                        total_correct += 1
                else:
                    female_total += 1
                    if pred_label == 'female':
                        female_correct += 1
                        total_correct += 1


    overall_accuracy = (total_correct / total_images) * 100 if total_images > 0 else 0
    male_accuracy = (male_correct / male_total) * 100 if male_total > 0 else 0
    female_accuracy = (female_correct / female_total) * 100 if female_total > 0 else 0

    print("\n✅ Hackathon Evaluation Results:")
    print(f"Total Images      : {total_images}")
    print(f"✔️  Correct        : {total_correct}")
    print(f"❌ Incorrect      : {total_images - total_correct}")
    print(f"✅ Overall Accuracy: {overall_accuracy:.2f}%\n")
    print(f" Male Accuracy    : {male_accuracy:.2f}% (Correct: {male_correct}/{male_total})")
    print(f" Female Accuracy  : {female_accuracy:.2f}% (Correct: {female_correct}/{female_total})")

def main():

    val_dir = "./hackathon_test_dataset/"

    model_path = "./hybrid_gender_model.h5"

   
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )
    print(" Model loaded successfully!")

    evaluate_unseen_dataset(model, val_dir, image_size=(224, 224))

if __name__ == "__main__":
    main()
