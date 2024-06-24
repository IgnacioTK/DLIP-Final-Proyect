############################### Detectar webcam 2 view + Modelo + String ###################################################
import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
from torchvision import datasets

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Definir las transformaciones necesarias (ajústalas según lo que usaste en el entrenamiento)
preprocess = transforms.Compose([
    transforms.Resize((200, 200)),  # Cambia esto al tamaño esperado por tu modelo
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Cambia esto según tu dataset
])

# Cargar el modelo guardado
model = torch.load('C:/Users/Ignacio/source/repos/DLIP/ProyectoFinalInternet/mejorModelo.pth')
model.eval()

# Obtener nombres de clases
data_dir = 'C:/Users/Ignacio/source/repos/DLIP/ProyectoFinalInternet/asl_alphabet_validation_detected'
dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
class_names = dataset.classes

# Verificar si hay una GPU disponible y usarla si es posible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Abrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Variables para la detección continua de la misma letra
tiempo_inicio = time.time()
letra_actual = None
letra_guardada = None
string_final = ''

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Crear una copia del frame original para la detección de la mano
    detection_frame = frame.copy()

    # Convertir el fotograma a RGB (MediaPipe Hands requiere imágenes en formato RGB)
    image_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Reiniciar el contador si no hay manos detectadas
    if not results.multi_hand_landmarks:
        tiempo_inicio = time.time()
        letra_actual = None
        letra_guardada = None
    else:
        # Si se detectan manos, dibujar los landmarks en el fotograma original y clasificar la postura
        for hand_landmarks in results.multi_hand_landmarks:
            # Dibujar los landmarks
            mp_drawing.draw_landmarks(detection_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer la región de la mano
            img_h, img_w, _ = frame.shape
            bbox = []
            for landmark in hand_landmarks.landmark:
                bbox.append((int(landmark.x * img_w), int(landmark.y * img_h)))
            bbox = np.array(bbox)

            # Obtener los límites de la caja
            x_min = np.min(bbox[:, 0])
            y_min = np.min(bbox[:, 1])
            x_max = np.max(bbox[:, 0])
            y_max = np.max(bbox[:, 1])

            # Expandir ligeramente la caja para capturar mejor la mano
            padding = 20
            x_min = max(x_min - padding, 0)
            y_min = max(y_min - padding, 0)
            x_max = min(x_max + padding, img_w)
            y_max = min(y_max + padding, img_h)

            # Recortar la región de la mano
            hand_image = frame[y_min:y_max, x_min:x_max]
            hand_image_pil = Image.fromarray(cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB))

            # Preprocesar la imagen de la mano
            input_tensor = preprocess(hand_image_pil)
            input_batch = input_tensor.unsqueeze(0).to(device)

            # Realizar la predicción
            with torch.no_grad():
                output = model(input_batch)
                _, predicted = torch.max(output, 1)
                predicted_class = predicted.item()
                predicted_label = class_names[predicted_class]

            # Mostrar la predicción en el fotograma
            cv2.putText(detection_frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Lógica para la detección continua de la misma letra
            if letra_actual != predicted_label:
                letra_actual = predicted_label
                tiempo_inicio = time.time()
            else:
                if time.time() - tiempo_inicio >= 1.5:
                    if letra_actual == 'space':
                        letra_guardada = letra_actual
                        string_final += ' '
                        tiempo_inicio = time.time()
                    elif letra_actual == 'del' and len(string_final) > 0:
                        letra_guardada = letra_actual
                        string_final = string_final[:-1]
                        tiempo_inicio = time.time()
                    else:
                        letra_guardada = letra_actual
                        string_final += letra_guardada
                        tiempo_inicio = time.time()

            # Mostrar la letra guardada en pantalla
            if letra_guardada:
                cv2.putText(detection_frame, f'Letra detectada: {letra_guardada}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.putText(detection_frame, f'String Final: {string_final}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Combinar el frame original y el frame con detección de mano
    combined_frame = np.hstack((frame, detection_frame))

    # Mostrar el fotograma combinado
    cv2.imshow('Original (Left) and Hand Detection (Right)', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()