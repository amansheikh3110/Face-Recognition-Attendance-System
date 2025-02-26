import cv2
import face_recognition
import numpy as np


# Load images and convert them from BGR (OpenCV format) to RGB
image_of_person1 = face_recognition.load_image_file('aman1.jpg')
image_of_person1 = face_recognition.load_image_file('aman2.jpg')
image_of_person1 = face_recognition.load_image_file('aman3.jpg')
image_of_person1 = face_recognition.load_image_file('aman4.jpg')
image_of_person2 = face_recognition.load_image_file('kashif1.jpg')
image_of_person2 = face_recognition.load_image_file('kashif2.jpg')
image_of_person3 = face_recognition.load_image_file('shafiya1.jpg')
image_of_person3 = face_recognition.load_image_file('shafiya2.jpg')
image_of_person4 = face_recognition.load_image_file('kamil1.jpg')

# Encode the faces in the images
person1_encoding = face_recognition.face_encodings(image_of_person1)[0]
person2_encoding = face_recognition.face_encodings(image_of_person2)[0]
person3_encoding = face_recognition.face_encodings(image_of_person3)[0]
person4_encoding = face_recognition.face_encodings(image_of_person4)[0]

# Create a list of known face encodings and their names
known_face_encodings = [
    person1_encoding,
    person2_encoding,
    person3_encoding,
    person4_encoding
]

known_face_names = [
    "Aman Sheikh",
    "Kashif Sheikh",
    "Shafiya Sheikh",
    "kamil Sheikh",
]
# Initialize the webcam
video_capture = cv2.VideoCapture(0)


while True:
    # Capture a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Initialize an array for the names of the detected faces
    face_names = []

    for face_encoding in face_encodings:
        # See if the face is a match for any known face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, use the first one
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # Release the webcam
video_capture.release()
cv2.destroyAllWindows()
