#base model Salesforce/blip-vqa-base
#pip install opencv-python
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
import cv2
def capture_image():
    # Open a connection to the default webcam (usually the first camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Read one frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read correctly
    if ret:
        # Save the captured image to a file
        cv2.imwrite('captured_image.jpg', frame)
        print("Image captured and saved as 'captured_image.jpg'.")
    else:
        print("Error: Could not read frame from webcam.")
        

    # Release the webcam and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Capture the image
capture_image()
print("OK")
raw_image = Image.open('C:/Users/Lenovo/Desktop/AWS/ML/captured_image.jpg').convert('RGB')

inputs = []
print("Enter your inputs (type 'exit' to finish):")
#user intraction
while True:
        question = input("Enter something: ")
        if question.lower() == 'exit':
            break
        inputs = processor(raw_image, question, return_tensors="pt")

        out = model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True))
