import cv2
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt") 

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run inference with stream=True for memory efficiency
    results = model.predict(frame, stream=True, conf=0.05)

    # Visualize results on the frame
    for r in results:
        annotated_frame = r.plot()

    # Display the output
    cv2.imshow("YOLO11 Real-Time Detection", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()