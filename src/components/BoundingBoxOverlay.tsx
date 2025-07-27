// src/components/BoundingBoxOverlay.tsx
import React from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Rect, Text as SvgText } from 'react-native-svg';
import { Detection, CLASS_LABELS } from '../utils/modelUtils'; // Import Detection type and labels

interface BoundingBoxOverlayProps {
  detections: Detection[];
  imageWidth: number;
  imageHeight: number;
  type?: 'static' | 'live'; // 'static' for uploaded images, 'live' for camera stream
}

const BoundingBoxOverlay: React.FC<BoundingBoxOverlayProps> = ({ detections, imageWidth, imageHeight, type = 'static' }) => {
  if (!detections || detections.length === 0) {
    return null;
  }

  return (
    <View style={[StyleSheet.absoluteFillObject, { width: imageWidth, height: imageHeight }]}>
      <Svg height={imageHeight} width={imageWidth}>
        {detections.map((detection, index) => {
          const [x1, y1, x2, y2] = detection.box;
          const label = CLASS_LABELS[Math.round(detection.classId)];
          const confidence = (detection.confidence * 100).toFixed(1);

          // Only draw boxes for 'person' class and with a minimum confidence
          if (label === 'person' && detection.confidence > 0.4) { // Adjustable confidence threshold
            const boxColor = type === 'live' ? 'lime' : 'red';
            const textColor = type === 'live' ? 'white' : 'white';

            return (
              <React.Fragment key={`${type}-${index}`}>
                <Rect
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  stroke={boxColor}
                  strokeWidth="2"
                  fill="none"
                />
                <SvgText
                  x={x1 + 2}
                  y={y1 < 15 ? 15 : y1 - 2} // Adjust y to prevent text going off-screen at top
                  fill={textColor}
                  fontSize="12"
                  fontWeight="bold"
                  stroke="black"
                  strokeWidth="0.5"
                >
                  {`${label} ${confidence}%`}
                </SvgText>
              </React.Fragment>
            );
          }
          return null;
        })}
      </Svg>
    </View>
  );
};

export default BoundingBoxOverlay;