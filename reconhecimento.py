import cv2
import face_recognition
import os

# Carrega as imagens e os nomes dos clientes
caminho_clientes = 'pics/'
imagens_clientes = []
nomes_clientes = []

for arquivo in os.listdir(caminho_clientes):
    img = face_recognition.load_image_file(caminho_clientes + arquivo)
    img_encoding = face_recognition.face_encodings(img)[0]
    imagens_clientes.append(img_encoding)
    nomes_clientes.append(os.path.splitext(arquivo)[0])

# Inicia a captura de vídeo
video_capture = cv2.VideoCapture(0)

while True:
    # Captura frame por frame
    ret, frame = video_capture.read()

    # Converte o frame de BGR para RGB (padrão usado pelo face_recognition)
    rgb_frame = frame[:, :, ::-1]

    # Localiza todos os rostos no frame atual
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compara os rostos no frame com os rostos conhecidos
        matches = face_recognition.compare_faces(imagens_clientes, face_encoding)
        name = "Desconhecido"

        # Se houver uma correspondência, use o primeiro rosto correspondente
        if True in matches:
            match_index = matches.index(True)
            name = nomes_clientes[match_index]

        # Desenha um retângulo ao redor do rosto e o nome da pessoa
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

    # Exibe o resultado
    cv2.imshow('Video', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura e fecha as janelas
video_capture.release()
cv2.destroyAllWindows()
