
export const CLASS_LABELS = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
  'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
];

interface FastTFLiteModel {
  run: (inputs: Float32Array[]) => Float32Array[];
}

export interface Detection {
  box: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  classId: number;
}

// Function to calculate Intersection Over Union (IOU)
function calculateIOU(box1: [number, number, number, number], box2: [number, number, number, number]): number {
  const [x1, y1, x2, y2] = box1;
  const [X1, Y1, X2, Y2] = box2;

  const xi1 = Math.max(x1, X1);
  const yi1 = Math.max(y1, Y1);
  const xi2 = Math.min(x2, X2);
  const yi2 = Math.min(y2, Y2);

  const interArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);

  const box1Area = (x2 - x1) * (y2 - y1);
  const box2Area = (X2 - X1) * (Y2 - Y1);

  return interArea / (box1Area + box2Area - interArea);
}

// Non-Maximum Suppression (NMS) implementation
function nonMaxSuppression(
  boxes: [number, number, number, number][],
  confidences: number[],
  iouThreshold: number
): { box: [number, number, number, number]; confidence: number; originalIndex: number }[] {
  if (boxes.length === 0) {
    return [];
  }

  const indices = confidences
    .map((conf, idx) => ({ conf, idx }))
    .sort((a, b) => b.conf - a.conf)
    .map(item => item.idx);

  const selected: { box: [number, number, number, number]; confidence: number; originalIndex: number }[] = [];
  const suppressed = new Array(boxes.length).fill(false);

  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i];
    if (suppressed[idx]) {
      continue;
    }

    selected.push({ box: boxes[idx], confidence: confidences[idx], originalIndex: idx });

    for (let j = i + 1; j < indices.length; j++) {
      const nextIdx = indices[j];
      if (suppressed[nextIdx]) {
        continue;
      }

      const iou = calculateIOU(boxes[idx], boxes[nextIdx]);
      if (iou > iouThreshold) {
        suppressed[nextIdx] = true;
      }
    }
  }
  return selected;
}

// Function to parse YOLOv8 TFLite output
// Assumes output tensor is [1, 84, 8400] and then flattened
// where 84 -> [bbox_x, bbox_y, bbox_w, bbox_h, obj_conf, class_conf_0, ..., class_conf_79]
export const parseYOLOv8Output = (
  outputTensor: Float32Array, // The flattened output from the TFLite model
  originalWidth: number,
  originalHeight: number,
  confidenceThreshold: number = 0.25,
  iouThreshold: number = 0.45,
  modelInputSize: number = 640 // YOLOv8 input size (e.g., 640x640)
): Detection[] => {
  const data = outputTensor;
  const numValuesPerPrediction = 84; // 4 bbox + 1 obj_conf + 80 class_confs
  const numPredictions = data.length / numValuesPerPrediction;

  const boxes: [number, number, number, number][] = [];
  const confidences: number[] = [];
  const classIds: number[] = [];

  for (let i = 0; i < numPredictions; i++) {
    const offset = i * numValuesPerPrediction;
    const predictionSlice = data.slice(offset, offset + numValuesPerPrediction);

    const [x_center_norm, y_center_norm, w_norm, h_norm, obj_conf, ...class_confs] = predictionSlice;

    const classId = class_confs.indexOf(Math.max(...class_confs));
    const classConfidence = class_confs[classId];
    const totalConfidence = obj_conf * classConfidence;

    // Filter by overall confidence and only keep 'person' detections
    // Assuming 'person' is the first class (index 0) in CLASS_LABELS
    if (totalConfidence > confidenceThreshold && CLASS_LABELS[classId] === 'person') {
      // Convert normalized YOLO coordinates to pixel coordinates in the original image's aspect ratio
      // YOLO coords are relative to model input size (640x640)
      // First, scale to 640x640 pixel coordinates
      const x_center_model = x_center_norm * modelInputSize;
      const y_center_model = y_center_norm * modelInputSize;
      const w_model = w_norm * modelInputSize;
      const h_model = h_norm * modelInputSize;

      // Calculate top-left and bottom-right in model input space
      let x1_model = x_center_model - w_model / 2;
      let y1_model = y_center_model - h_model / 2;
      let x2_model = x_center_model + w_model / 2;
      let y2_model = y_center_model + h_norm / 2;


      // Now, scale these pixel coordinates from 640x640 to the original image's dimensions
      const scaleX = originalWidth / modelInputSize;
      const scaleY = originalHeight / modelInputSize;

      const x1_orig = x1_model * scaleX;
      const y1_orig = y1_model * scaleY;
      const x2_orig = x2_model * scaleX;
      const y2_orig = y2_model * scaleY;

      boxes.push([x1_orig, y1_orig, x2_orig, y2_orig]);
      confidences.push(totalConfidence);
      classIds.push(classId);
    }
  }

  // Apply NMS
  const filteredIndices = nonMaxSuppression(boxes, confidences, iouThreshold);

  // Reconstruct final detections with original classId
  const finalDetections: Detection[] = filteredIndices.map(item => ({
    box: item.box,
    confidence: item.confidence,
    classId: classIds[item.originalIndex], // Get the corresponding classId
  }));

  return finalDetections;
};

// Main inference function for static images
export const performInference = async (
  model: FastTFLiteModel | null,
  inputBuffer: Float32Array, // Assumed to be 640x640x3 Float32Array
  originalWidth: number,
  originalHeight: number
): Promise<{ detections: Detection[]; count: number }> => {
  try {
    if (!model) {
      console.error("AI Model not loaded for inference.");
      return { detections: [], count: 0 };
    }

    const output = await model.run([inputBuffer]);
    const outputTensor = output[0] as Float32Array; // Cast to Float32Array

    const detections = parseYOLOv8Output(outputTensor, originalWidth, originalHeight);
    const personDetections = detections.filter(d => CLASS_LABELS[Math.round(d.classId)] === 'person');
    const crowdCount = personDetections.length;

    return { detections: personDetections, count: crowdCount };
  } catch (error) {
    console.error("Error during inference:", error);
    throw error;
  }
};