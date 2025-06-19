import os

def embedding_mapper(images_folder):
    images = os.listdir(images_folder)
    embeddings = dict()
    for image in images:
        im = os.listdir(f"{images_folder}/{image}")
        for i in im:
            im = cv2.imread(os.path.join(images_folder, image, i))
            faces = model.get(im)
            if len(faces) > 0:
                for face in faces:
                    box = face.bbox
                    if image in embeddings.keys():
                        embeddings[image].append(face.embedding.tolist())
                    else:
                        embeddings[image] = [face.embedding.tolist()]
            else:
                continue
    with open("embeddings_mapper.json", "w") as f:
        j = json.dump(embeddings, f, indent=4)
