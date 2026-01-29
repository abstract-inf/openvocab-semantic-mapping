import sys
import cv2

# --- UPDATED MONKEY PATCH FOR TRUNCATE ARGUMENT ---
try:
    import open_clip
    import torch
    
    class FakeClip:
        def __init__(self):
            # Using the same model name ultralytics defaults to
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
        def load(self, name, device="cuda" if torch.cuda.is_available() else "cpu"):
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained='openai')
            return model.to(device), preprocess
            
        def tokenize(self, texts, truncate=True):
            # Added 'truncate' support to fix the TypeError
            return self.tokenizer(texts)

    # Inject the patch into sys.modules
    sys.modules['clip'] = FakeClip()
    
except ImportError:
    print("Run: pip install open_clip_torch torch")
    sys.exit()
# ------------------------------------------------

from ultralytics import YOLOWorld

# 1. Load the model
model = YOLOWorld("yolov8s-worldv2.pt") 

# 2. User Input
target = input("Object to find (e.g., door): ")
if not target: target = "person" # Default
model.set_classes([target])

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 3. Predict
    results = model.predict(frame, conf=0.05, verbose=False)
    r = results[0]

    # 4. Labeling Logic
    if len(r.boxes) > 0:
        frame = r.plot() # Draws bounding boxes
    else:
        # NO DETECTION: Indicator in top left
        cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
        cv2.putText(frame, f"{target.upper()} NOT FOUND", (50, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Open-Vocab Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()