def boundbox(image, box, text):
    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(0, 255, 0), 2)
    cv2.putText(image, text.replace("_", " "), (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)