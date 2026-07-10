
# Development Plan: Client-Side EMNIST Character Recognition Web App

## Phase 1: Model Training & Conversion (Python Environment)
The objective of this phase is to train the required Convolutional Neural Network (CNN) architecture and export it into a format that web browsers can execute natively.

* **Step 1.1: Train the CNN**
    * [cite_start]Load the `balanced` split of the EMNIST dataset (47 classes)[cite: 38, 81].
    * [cite_start]Implement the exact sequential architecture required[cite: 59, 61]:
        * [cite_start]Input layer: `(28, 28, 1)`[cite: 63, 73].
        * [cite_start]Block 1: `Conv2D` (32 filters, 5x5 kernel, `tanh`, `same` padding) + `MaxPooling2D` (stride = 2)[cite: 62, 63, 64].
        * [cite_start]Block 2: `Conv2D` (48 filters, 5x5 kernel, `tanh`, `same` padding) + `MaxPooling2D` (stride = 2)[cite: 65, 66].
        * [cite_start]Block 3: `Conv2D` (64 filters, 5x5 kernel, `tanh`, `same` padding) + `Flatten()`[cite: 67, 68].
        * [cite_start]Dense Layers: `Dense(512, tanh)` -> `Dense(84, tanh)` -> `Dense(47, softmax)`[cite: 72].
* **Step 1.2: Model Evaluation & Export**
    * [cite_start]Compute evaluation metrics: Accuracy, Precision, Recall, and the Confusion Matrix[cite: 84, 85, 87, 89, 91].
    * Save the trained model native Keras format: `model.save('emnist_model.keras')`.
* **Step 1.3: TensorFlow.js Conversion**
    * Install the conversion tool: `pip install tensorflowjs`.
    * Convert the model to a web-friendly format via command line:
        ```bash
        tensorflowjs_converter --input_format=keras emnist_model.keras ./web_model
        ```
    * Output: This generates a `model.json` file (architecture) and one or more `.bin` files (sharded weights).

## Phase 2: Project Structure (Web Frontend)
Organize your web repository as a static website. Since all processing runs on the client-side, no backend server is needed.

```text
emnist-web-app/
│
├── index.html          # UI Layout & Script CDNs
├── style.css           # Styling for Canvas, Buttons, and Layout
├── app.js              # Core Application Logic (Canvas, OpenCV.js, TF.js)
└── model/              # Converted Model Assets
    ├── model.json
    └── group1-shard1of1.bin

```

## Phase 3: UI Design & Canvas Interactivity (HTML/CSS/JS)

Build the frontend user interface to capture handwritten input.

* **Step 3.1: HTML Framework (`index.html`)**
* Add an HTML5 `<canvas>` element to act as the drawing surface.


* Add controls: a "Recognize Word" button and a "Clear Canvas" button.
* Add an output display text element to present the reconstructed word.


* Import dependencies via CDN in the `<head>` or bottom of `<body>`:
```html
<script src="[https://cdn.jsdelivr.net/npm/@tensorflow/tfjs](https://cdn.jsdelivr.net/npm/@tensorflow/tfjs)"></script>
<script src="[https://docs.opencv.org/4.x/opencv.js](https://docs.opencv.org/4.x/opencv.js)"></script>

```




* **Step 3.2: Drawing Logic (`app.js`)**
* Configure the canvas background to Black and the drawing stroke to White to match EMNIST dataset characteristics.
* Implement event listeners (`mousedown`, `mousemove`, `mouseup` / `touchstart`, `touchmove`, `touchend`) to track and draw smooth lines on the canvas context.



## Phase 4: Image Segmentation (OpenCV.js)

Translate the Python OpenCV sequence given in the challenge instructions into JavaScript.

* **Step 4.1: Convert Canvas to OpenCV Matrix**
* `let src = cv.imread(canvasElement);`


* **Step 4.2: Pre-processing Pipeline**
* Convert image to grayscale: `cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);`.


* Apply Gaussian Blur to smooth noise: `cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);`.


* Apply Inverted Otsu Thresholding: `cv.threshold(blur, thresh, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);`.




* **Step 4.3: Contours & Spatial Sorting**
* Find character contours: `cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);`.


* Loop through contours to extract Bounding Boxes `(x, y, w, h)`. Filter out micro-noise instances if needed.


* Sort the bounding box objects from left to right using their `x` coordinate value (`boxes.sort((a, b) => a.x - b.x);`) to guarantee correct reading order.





## Phase 5: Model Integration & Client-Side Inference (TensorFlow.js)

Load the converted neural network and run predictions iteratively for each segmented letter.

* **Step 5.1: Model Loading**
* Load the asynchronous model on window initialization:
```javascript
const model = await tf.loadLayersModel('model/model.json');

```




* **Step 5.2: Character Extraction & Pre-processing Loop**
* Iterate through each sorted bounding box:


* Crop the region of interest (ROI) from the threshold matrix using OpenCV.js.


* Resize the cropped character to exactly 28x28 pixels (`cv.resize`).


* Convert the processed OpenCV region into a TensorFlow.js Tensor (ensure 1-channel grayscale output).
* Normalize pixel intensities to a `[0, 1]` floating-point range: `tensor = tensor.toFloat().div(255.0);`.


* Reshape the tensor to fulfill the 4D input requirements `[1, 28, 28, 1]`.






* **Step 5.3: Word Reconstruction**
* Run prediction: `const prediction = model.predict(tensor);`.


* Obtain the highest probability index: `const classIndex = prediction.argMax(1).dataSync()[0];`.


* Map the index to your 47-character string array mapping.


* Append each identified character sequentially to assemble the complete string and render it onto the screen.





## Phase 6: Static Deploy

Since this runtime model relies exclusively on front-end assets, hosting does not require complex Python virtual environments.

* 
**Hugging Face Spaces Option:** Create a new Space, choose Static HTML as your SDK type, and commit your static assets (`index.html`, `style.css`, `app.js`, and the `/model` directory) directly.


* **Alternative Option (GitHub Pages):** Push the project repository onto GitHub and enable GitHub Pages under settings to instantly distribute a free public access URL.

