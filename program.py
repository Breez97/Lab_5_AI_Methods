import os, sys, random
import pickle
import face_recognition
import numpy as np

# Обучение модели
def train_model():
    if not os.path.exists("dataset"):
        print("There is no directory 'dataset'")
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

                    known_encodings.append(face_encoding)
                    known_names.append(person)

    data = {
        "names": known_names,
        "encodings": known_encodings
    }

    with open('encodings.pkl', 'wb') as file:
        file.write(pickle.dumps(data))

    return "============================================================\n\nFile encodings.pkl successfully created"


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
    best_matches = {}
    for (i, face_encoding) in enumerate(face_encodings):
        distances = face_recognition.face_distance(encodings, face_encoding)

        print(f"\n\033[95mProcessing face:\033[0m")
        for j, distance in enumerate(distances):
            confidence = 1 - distance
            if names[j] not in best_matches or confidence > best_matches[names[j]]["confidence"]:
                best_matches[names[j]] = {"confidence": confidence, "index": j}

    best_confidence = max((match["confidence"] for match in best_matches.values()))
    for name, result in best_matches.items():
        label = ""
        if result["confidence"] == best_confidence and result['confidence'] > 0.8:
            label = "\033[92m <<< Best >>> \033[0m"
        print(f"\033[94m\tPerson:\033[0m {name} \033[96m: coincidence: {result['confidence']:.4f}\033[0m {label}")


def main():
    print(train_model())
    names, encodings = load_encodings()
    images_folder = "images"
    for image_filename in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_filename)
        print(f"\n============================================================\n\n\033[95mProcessing image:\033[0m {image_path}")
        compare_faces(image_path, encodings, names)

if __name__ == '__main__':
    main()
