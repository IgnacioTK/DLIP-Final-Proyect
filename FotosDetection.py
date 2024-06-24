import cv2
import mediapipe as mp
import os
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Ruta de la carpeta original con las imágenes
dataset_folder = '/home/DLIP_user3/Proyect/asl_alphabet_train'
# Ruta de la carpeta donde se guardarán las imágenes procesadas
output_folder = '/home/DLIP_user3/Proyect/asl_alphabet_train_detected'

# Función para intentar detectar manos ajustando alpha y beta
def detect_hands_with_first_valid_adjustment(image):
    alpha_values = np.arange(0.5, 2.1, 0.1)  # Valores de alpha de 0.5 a 2 en pasos de 0.1
    beta_values = np.arange(0, 121, 5)      # Valores de beta de 0 a 120 en pasos de 5

    for alpha in alpha_values:
        for beta in beta_values:
            adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            image_rgb = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                return adjusted_image, results

    return None, None

# Crear la estructura de carpetas en la carpeta de salida
for letter_folder in os.listdir(dataset_folder):
    input_letter_folder_path = os.path.join(dataset_folder, letter_folder)
    output_letter_folder_path = os.path.join(output_folder, letter_folder)
    
    if not os.path.isdir(input_letter_folder_path):
        continue  # Ignorar si no es una carpeta
    
    os.makedirs(output_letter_folder_path, exist_ok=True)

# Procesar cada imagen en el dataset
for letter_folder in os.listdir(dataset_folder):
    input_letter_folder_path = os.path.join(dataset_folder, letter_folder)
    output_letter_folder_path = os.path.join(output_folder, letter_folder)

    if not os.path.isdir(input_letter_folder_path):
        continue  # Ignorar si no es una carpeta

    for image_file in os.listdir(input_letter_folder_path):
        image_path = os.path.join(input_letter_folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: No se pudo cargar la imagen {image_file}.")
            continue

        adjusted_image, results = detect_hands_with_first_valid_adjustment(image)

        if results is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(adjusted_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Guardar la imagen procesada en la carpeta de salida
            output_image_path = os.path.join(output_letter_folder_path, image_file)
            cv2.imwrite(output_image_path, adjusted_image)
            print(f"Imagen guardada: {output_image_path}")
        else:
            print(f"No se detectaron manos en la imagen {image_file}.")
