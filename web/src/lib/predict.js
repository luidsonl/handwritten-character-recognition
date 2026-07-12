import * as tf from '@tensorflow/tfjs';
import { LABELS } from './labels';

export function segmentAndPredict(canvas, model) {
  if (!window.opencvReady || typeof cv === 'undefined' || typeof cv.imread !== 'function') {
    throw new Error('OpenCV.js ainda nao esta pronto');
  }

  const src = cv.imread(canvas);
  if (!src || src.empty()) {
    if (src) src.delete();
    throw new Error('Falha ao ler canvas');
  }

  const gray = new cv.Mat();
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

  const blurred = new cv.Mat();
  cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

  const thresh = new cv.Mat();
  cv.threshold(blurred, thresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

  const boxes = [];
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const rect = cv.boundingRect(contour);
    const area = rect.width * rect.height;
    if (area > 50 && rect.width > 3 && rect.height > 3) {
      boxes.push({ x: rect.x, y: rect.y, w: rect.width, h: rect.height });
    }
    contour.delete();
  }

  boxes.sort((a, b) => a.x - b.x);

  const predictions = [];
  for (const box of boxes) {
    const padding = Math.max(Math.floor(Math.max(box.w, box.h) * 0.15), 4);
    const sx = Math.max(0, box.x - padding);
    const sy = Math.max(0, box.y - padding);
    const sw = Math.min(src.cols - sx, box.w + padding * 2);
    const sh = Math.min(src.rows - sy, box.h + padding * 2);

    const roiRect = new cv.Rect(sx, sy, sw, sh);
    const roi = gray.roi(roiRect);

    const rotated = new cv.Mat();
    cv.rotate(roi, rotated, cv.ROTATE_90_CLOCKWISE);
    const flipped = new cv.Mat();
    cv.flip(rotated, flipped, 1);

    const square = new cv.Mat();
    const side = Math.max(flipped.cols, flipped.rows);
    const padX = Math.floor((side - flipped.cols) / 2);
    const padY = Math.floor((side - flipped.rows) / 2);
    cv.copyMakeBorder(flipped, square, padY, side - flipped.rows - padY, padX, side - flipped.cols - padX, cv.BORDER_CONSTANT, new cv.Scalar(0));

    const resized = new cv.Mat();
    cv.resize(square, resized, new cv.Size(28, 28), 0, 0, cv.INTER_AREA);

    const data = resized.data;
    const input = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
      input[i] = data[i] / 255.0;
    }
    const tensor = tf.tensor4d(input, [1, 28, 28, 1]);

    const prediction = model.predict(tensor);
    const scores = prediction.dataSync();
    const classIndex = prediction.argMax(1).dataSync()[0];
    const confidence = scores[classIndex];

    predictions.push({ char: LABELS[classIndex], confidence, box });

    tensor.dispose();
    prediction.dispose();
    resized.delete();
    square.delete();
    flipped.delete();
    rotated.delete();
    roi.delete();
  }

  src.delete();
  gray.delete();
  blurred.delete();
  thresh.delete();
  contours.delete();
  hierarchy.delete();

  return predictions;
}
