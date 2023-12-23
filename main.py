import pickle
import face_recognition
import os
import sys
import numpy as np

# Обучение модели
def train_model():
    if not os.path.exists("dataset"):
        print("[ERROR] there is no directory 'dataset'")
        sys.exit()

    known_encodings = []
    known_names = []

    persons = os.listdir("dataset")

    for person in persons:
        person_path = os.path.join("dataset", person)
        if os.path.isdir(person_path):
            images = os.listdir(person_path)

            for (i, image) in enumerate(images):
                image_path = os.path.join(person_path, image)

                face_img = face_recognition.load_image_file(image_path)
                face_encodings = face_recognition.face_encodings(face_img)

                if len(face_encodings) > 0:
                    face_encoding = face_encodings[0]

                    match_found = False
                    for enc in known_encodings:
                        if len(enc) > 0:
                            distance = np.linalg.norm(face_encoding - enc)
                            if distance <= tolerance:
                                match_found = True
                                break

                    if not match_found:
                        known_encodings.append(face_encoding)
                        known_names.append(person)

    data = {
        "names": known_names,
        "encodings": known_encodings
    }

    with open('encodings.pkl', 'wb') as file:
        file.write(pickle.dumps(data))

    return "============================================================\n\n[INFO] File encodings.pkl successfully created"

# Загрузка encoding, созданного на основе датасета
def load_encodings(filename="encodings.pkl"):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data["names"], data["encodings"]

# Сравнение лиц
def compare_faces(image_path, encodings, names):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (i, face_encoding) in enumerate(face_encodings):
        distances = face_recognition.face_distance(encodings, face_encoding)

        print(f"\nProcessing face:")
        for j, distance in enumerate(distances):
            confidence = 1 - distance
            print(f"  Person: {names[j]}, Result: {confidence:.4f}")

def main():
    print(train_model())

    names, encodings = load_encodings()

    images_folder = "images"
    for image_filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_filename)
        print(f"\n============================================================\n\nProcessing image: {image_path}")
        compare_faces(image_path, encodings, names)

if __name__ == '__main__':
    tolerance = 0.75
    main()
