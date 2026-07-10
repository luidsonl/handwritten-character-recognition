
# Development Plan: Client-Side EMNIST Character Recognition Web App

> A proof-of-concept (POC) system for intelligent handwritten word recognition.
> The user writes a word on a canvas and the system reconstructs it as digital text.

---

## Project Structure

```text
handwritten-character-recognition/
│
├── training/                    # Model training (Jupyter Notebook)
│   ├── main.ipynb               # Full training pipeline + evaluation
│   ├── requirements.txt         # Python dependencies
│   └── .gitignore               # Ignores venv/, __pycache__/, *.keras, web_model/
│
├── web/                         # Static web application (deployed)
│   ├── index.html               # UI layout & CDN imports
│   ├── style.css                # Styling for canvas, buttons, layout
│   ├── app.js                   # Core logic: Canvas, OpenCV.js, TF.js
│   └── model/                   # Converted model assets
│       ├── model.json           # TF.js model architecture
│       └── group1-shard1of1.bin # Trained weights (binary)
│
├── docs/
│   ├── development-plan.md      # This file
│   └── (reports, diagrams)
│
├── .gitignore                   # Root gitignore
└── README.md                    # Project overview + public access link
```

---

## Phase 1: Model Training & Evaluation (Python)

**Goal:** Train the CNN on EMNIST, evaluate performance, export for web use.

### Step 1.1: Environment Setup

- Create and activate a Python virtual environment.
- Install dependencies from `training/requirements.txt`.
- Key libraries: `tensorflow`, `emnist`, `opencv-python`, `scikit-learn`, `matplotlib`, `numpy`.

### Step 1.2: Load EMNIST Dataset

- Use the **`balanced`** split of the EMNIST dataset (47 classes).
- The `emnist` Python library provides direct access.
- Load training and test sets, reshape images to `(28, 28, 1)`, normalize pixel values to `[0, 1]` by dividing by 255.0.
- Apply any necessary transposing/flipping to correct EMNIST image orientation.

### Step 1.3: Implement the CNN Architecture (Sequential API)

The model **must** follow this exact topology:

```
Input: (28, 28, 1) — grayscale

Block 1:
  Conv2D(32, kernel=5x5, padding='same', activation='tanh')
  MaxPooling2D(pool_size=2, strides=2)

Block 2:
  Conv2D(48, kernel=5x5, padding='same', activation='tanh')
  MaxPooling2D(pool_size=2, strides=2)

Block 3:
  Conv2D(64, kernel=5x5, padding='same', activation='tanh')

Flatten

Dense(512, activation='tanh')
Dense(84, activation='tanh')
Dense(47, activation='softmax')
```

- Compile with an appropriate optimizer (e.g., `adam`) and `categorical_crossentropy` loss.
- Train for a sufficient number of epochs with validation split.

### Step 1.4: Evaluation & Metrics

After training, **mandatory** evaluation on the test set:

| Metric             | Tool                        |
|--------------------|-----------------------------|
| Accuracy           | `model.evaluate()`          |
| Precision          | `sklearn.metrics.precision_score()` |
| Recall             | `sklearn.metrics.recall_score()`    |
| Confusion Matrix   | `sklearn.metrics.confusion_matrix()` |

### Step 1.5: Theoretical Analysis (Report)

Answer the following in the notebook (Markdown cells):

1. **Which characters have the highest confusion index?**
   - Analyze the confusion matrix to identify the most misclassified class pairs.

2. **What theoretical hypotheses justify these classification difficulties?**
   - Consider visual similarity between characters (e.g., 'l' vs '1', 'O' vs '0', 'I' vs 'l').
   - Consider class imbalance in the EMNIST balanced split.
   - Consider stroke variation and handwriting style differences.

### Step 1.6: Model Export

- Save in native Keras format: `model.save('emnist_model.keras')`.
- Convert to TensorFlow.js format:
  ```bash
  pip install tensorflowjs
  tensorflowjs_converter --input_format=keras emnist_model.keras ./web_model
  ```
- Output: `model.json` + `group1-shard1of1.bin` (or multiple shards).
- Copy the `web_model/` contents into `web/model/` for deployment.

---

## Phase 2: Static Web Frontend

**Goal:** Build the client-side application. No backend server required.

### Step 2.1: HTML Structure (`web/index.html`)

- HTML5 `<canvas>` element as the drawing surface.
- Controls: "Recognize Word" button, "Clear Canvas" button.
- Output display element for the reconstructed word.
- CDN imports in `<head>`:
  ```html
  <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
  <script src="https://docs.opencv.org/4.x/opencv.js"></script>
  ```

### Step 2.2: Styling (`web/style.css`)

- Responsive layout.
- Canvas styling (border, cursor, dimensions).
- Button and output text styling.

### Step 2.3: Drawing Logic (`web/app.js`)

- Canvas background: **black**. Drawing stroke: **white** (matches EMNIST).
- Event listeners for mouse: `mousedown`, `mousemove`, `mouseup`.
- Event listeners for touch: `touchstart`, `touchmove`, `touchend`.
- Smooth line drawing using canvas context (`lineWidth`, `lineCap`, `lineJoin`).

---

## Phase 3: Image Segmentation (OpenCV.js)

**Goal:** Extract individual characters from the drawn word.

### Step 3.1: Canvas to OpenCV Matrix

```javascript
let src = cv.imread(canvasElement);
```

### Step 3.2: Pre-processing Pipeline

1. Convert to grayscale:
   ```javascript
   cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
   ```
2. Gaussian blur (5x5) for noise reduction:
   ```javascript
   cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
   ```
3. Inverted Otsu thresholding:
   ```javascript
   cv.threshold(blur, thresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
   ```

### Step 3.3: Contour Detection & Spatial Sorting

1. Find external contours:
   ```javascript
   cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
   ```
2. Extract bounding boxes `(x, y, w, h)` from each contour.
3. Filter out noise (e.g., boxes smaller than a minimum area threshold).
4. Sort boxes **left to right** by x-coordinate:
   ```javascript
   boxes.sort((a, b) => a.x - b.x);
   ```

---

## Phase 4: Model Integration & Inference (TensorFlow.js)

**Goal:** Run predictions on each segmented character.

### Step 4.1: Load Model

```javascript
const model = await tf.loadLayersModel('model/model.json');
```

### Step 4.2: Character Extraction & Prediction Loop

For each sorted bounding box:

1. Crop the Region of Interest (ROI) from the thresholded image.
2. Resize to **28x28** pixels (`cv.resize`).
3. Convert OpenCV matrix to TensorFlow.js tensor (single channel).
4. Normalize to `[0, 1]`: `tensor.toFloat().div(255.0)`.
5. Reshape to 4D: `[1, 28, 28, 1]`.
6. Run prediction:
   ```javascript
   const prediction = model.predict(tensor);
   const classIndex = prediction.argMax(1).dataSync()[0];
   ```
7. Map the index to the 47-character EMNIST balanced class mapping.

### Step 4.3: Word Reconstruction

- Append each predicted character to a string.
- Display the final word in the output text element.

---

## Phase 5: Testing & Validation

**Goal:** Verify end-to-end functionality.

### Step 5.1: Unit Testing

- Test canvas drawing and clearing.
- Test OpenCV preprocessing pipeline (grayscale, blur, threshold, contours).
- Test model loading and single-character prediction.

### Step 5.2: Integration Testing

- Write complete words on the canvas and verify correct recognition.
- Test edge cases: closely spaced letters, large gaps, varying stroke widths.

### Step 5.3: Cross-browser Testing

- Verify functionality on Chrome, Firefox, Safari.
- Test touch input on mobile devices.

---

## Phase 6: Deployment

**Goal:** Make the application publicly accessible.

### Step 6.1: Repository Preparation

- Ensure `web/model/` contains the converted TF.js model files.
- Update `.gitignore` to exclude training artifacts (`venv/`, `__pycache__/`, `*.keras`, `training/web_model/`).
- Write a clear `README.md` with project description and access link.

### Step 6.2: Deployment Options

| Platform       | Type              | Notes                                      |
|----------------|-------------------|---------------------------------------------|
| GitHub Pages   | Static hosting    | Enable in repo Settings > Pages             |
| Hugging Face   | Static HTML Space | Upload `web/` contents directly             |

### Step 6.3: Final Submission

- Public URL accessible to the professor.
- Notebook (`.ipynb`) with full training, evaluation, and theoretical analysis.
- All code organized and documented.

---

## Checklist

- [ ] EMNIST balanced dataset loaded (47 classes)
- [ ] CNN architecture matches required topology exactly
- [ ] Model trained with sufficient epochs
- [ ] Accuracy, Precision, Recall computed
- [ ] Confusion Matrix generated and analyzed
- [ ] Theoretical questions answered in notebook
- [ ] Model exported to TensorFlow.js format
- [ ] Web frontend with canvas, buttons, output display
- [ ] OpenCV.js segmentation pipeline working
- [ ] TF.js model loads and predicts in browser
- [ ] End-to-end word recognition tested
- [ ] Deployed and public URL available
- [ ] Repository clean with proper `.gitignore`
