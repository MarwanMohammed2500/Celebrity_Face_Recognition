# Celebrity Face Recognition with InsightFace (AntelopeV2)

This project performs **celebrity face recognition** in video using **InsightFace's AntelopeV2** model. It combines face detection, embedding generation, and face tracking to annotate celebrities' faces in video footage.

### Features

- Scrapes celebrity face images using Bing
- Extracts embeddings using InsightFace's `antelopev2`
- Matches faces based on cosine similarity
- Uses Deep SORT to track individuals across frames
- Annotates and exports labeled video
- Modular codebase (scraper, mapper, matcher, face_recognizer, comparator, and bbox)
- Testable via Jupyter notebook

---

## Project Structure

```

.
├── bbox.py                 # Utility for drawing bounding boxes
├── comparator.py           # Calculates cosine similarity
├── face_recognizer.py      # Main script to process and annotate video
├── mapper.py               # Maps embeddings from scraped images
├── matcher.py              # Matches query embedding to known identities
├── scrapper.py             # Downloads celebrity images using Bing
├── embeddings_mapper.json  # Generated after mapping known faces
├── video_input/            # Input videos
├── video_output/           # Output videos with annotations
├── celeb_images/           # Scraped images of celebrities
├── celeb.ipynb             # Jupyter notebook for experimentation

````
**NOTE:**
I didn't upload any images or videos for a few reasons, but you can still use the scraper to get the images, and use whatever videos you like!

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
````

**Main packages:**

* `insightface`
* `opencv-python`
* `numpy`
* `deep_sort_realtime`
* `icrawler`

---

## How It Works

1. **Image Collection**

   * Uses `icrawler` to scrape face images of celebrities.
   * See: `scrapper.py`

2. **Embedding Generation**

   * Loads each image and generates embeddings using `antelopev2`.
   * Stores them in `embeddings_mapper.json`.
   * See: `mapper.py`

3. **Face Recognition in Video**

   * Detects faces, tracks them with Deep SORT.
   * Matches each face with known embeddings using cosine similarity.
   * Annotates faces and exports the labeled video.
   * See: `face_recognizer.py`

---

## Example Output

A video of Elon Musk is analyzed and annotated with bounding boxes and names of detected celebrities.

> Example output saved to: `video_output/elon_musk_output_labeled.mp4`

---

## Testing via Notebook

A `celeb.ipynb` file is included to experiment with the components individually and test recognition on images interactively.

---

## Customization

* **To add new celebrities**, more names are present in `celebrity_list.csv`, you can check it, add or remove names, etc.
* **Adjust similarity threshold** in `matcher.py` (default: `0.3`).
* **Change video source** in `face_recognizer.py`.

---

## Notes

* The system only performs recognition on frames sampled every half second (`fps / 2`).
* Detected faces without confident matches are labeled as `"UNKNOWN"`.
* All scraped images are used for embedding without manual filtering, so quality may affect performance.

---

## License

This project is for educational/demo purposes. Ensure compliance with any third-party terms when using scraped celebrity data.

---

## Contributions

Pull requests, feedback, and improvements are welcome!

```
