import cv2

# Load video and classifiers
video = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smileCascade = cv2.CascadeClassifier("haarcascade_smile.xml")

while True:
    ret, image = video.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Region of interest for smile detection within the face
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        smiles = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=20
        )

        for (sx, sy, sw, sh) in smiles:
            # Draw rectangle around smile
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)

            # Save the image only once per smile detection
            cv2.imwrite("smile_captured.jpg", image)
            print("Smile Detected and Image Saved")
            break  # Exit smile loop to avoid multiple saves

    cv2.imshow('Live Video - Press q to Quit', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()