#!/usr/bin/python
from __future__ import print_function
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from httplib2 import Http
from oauth2client import file, client, tools
from PIL import Image, ImageDraw

import face_recognition
import io
import cv2
import dlib


# If modifying these scopes, delete the file token.json.
SCOPES = 'https://www.googleapis.com/auth/drive'

def main():
    store = file.Storage('token.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('credentials.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('drive', 'v3', http=creds.authorize(Http()))

    # Call the Drive v3 API
    file_id = '1ALXmFWrnaV7vhCDMPW2N4sPudTsqpjr-'
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO('nicolascage.jpg', 'w')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print ("Download %d%%." % int(status.progress() * 100))
        

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file("nicolascage.jpg")

    # Find all the faces in the image using a pre-trained convolutional neural network.
    # This method is more accurate than the default HOG model, but it's slower
    # unless you have an nvidia GPU and dlib compiled with CUDA extensions. But if you do,
    # this will use GPU acceleration and perform well.
    # See also: find_faces_in_picture.py
    face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")
    
    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    for face_landmarks in face_landmarks_list:
        d = ImageDraw.Draw(pil_image, 'RGBA')

        # Make the eyebrows into a nightmare
        d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
        d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
        d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
        d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

        # Gloss the lips
        d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
        d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
        d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
        d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

        # Sparkle the eyes
        d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
        d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

        # Apply some eyeliner
        d.line(face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
        d.line(face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

        print("I found {} face(s) in this photograph.".format(len(face_locations)))

        for face_location in face_locations:

            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)

        #Save photo and display
        pil_image.save("nicolascage.jpg")
        pil_image.show()

if __name__ == '__main__':
    main()