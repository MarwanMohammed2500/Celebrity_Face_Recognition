import json
import numpy as np
from insightface.app import FaceAnalysis
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import pickle as pkl
from matcher import match_face
from bbox import boundbox

# Open the embeddings file
with open("embeddings_mapper.json", "r") as f:
    loaded_emb = json.load(f)

# Prepare the model
model = FaceAnalysis(name='antelopev2',
                     download=True,
                     root="./",
                     providers=['CPUExecutionProvider'])
model.prepare(ctx_id=-1)

# Face Tracker
tracker = DeepSort(max_age=30)

# Prepare the video
cap = cv2.VideoCapture('video_input/elon_musk.mp4')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_index = 0
res = dict()
frame_data = defaultdict(list)

# Process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # print("NOT RET")
        break

    detections = []
    embeddings = []
    box_label_pair = []

    if frame_index%(int(fps/2)) == 0:
        faces = model.get(frame)
        for face in faces:
            box = face.bbox.astype(int)
            emb = face.embedding
            label = match_face(emb, loaded_emb)
            detections.append(([box[0], box[1], box[2], box[3]], 0.99, 'face'))
            embeddings.append(emb)
            box_label_pair.append((box, label[0]))
        
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id

            for bbox, lab in box_label_pair:
                x1, y1, x2, y2 = bbox
                best_label = lab or "UNKOWN"
            
                frame_data[frame_index].append({
                    "track_id": track_id,
                    "bbox": bbox,
                    "label": best_label
        })
        frame_index += 1
        frame_data[frame_index] = frame_data[frame_index-1]
    else:
        frame_data[frame_index] = frame_data[frame_index-1]
        frame_index += 1
cap.release()

# Render the video after annotation
cap = cv2.VideoCapture("video_input/elon_musk.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("video_output/elon_musk_output_labeled.mp4", fourcc, fps, (width, height))

frame_index = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index in frame_data:
        for item in frame_data[frame_index]:
            l, t, r, b = item["bbox"]
            label = item["label"]
            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

    out.write(frame)
    cv2.imshow("Annotated", frame) # Show the rendered video
    if cv2.waitKey(round(1000 / fps)) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
out.release()