// src/screens/CameraScreen.tsx
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { View, Text, StyleSheet, Dimensions, Platform, PermissionsAndroid, Alert } from 'react-native';
import { Appbar, useTheme, Snackbar } from 'react-native-paper';
import { Camera, useCameraDevices, useFrameProcessor, CameraDevice } from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite'; // Removed 'Model' from import
import Animated, { useSharedValue, useAnimatedStyle, withSpring, runOnJS } from 'react-native-reanimated';
import { useFocusEffect } from '@react-navigation/native';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';
import { parseYOLOv8Output, CLASS_LABELS, Detection } from '../utils/modelUtils';
import { processCameraFrame } from '../utils/imageProcessing';

const { width, height } = Dimensions.get('window');

const CameraScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const theme = useTheme();

  const [hasPermission, setHasPermission] = useState<boolean>(false);
  const [crowdCount, setCrowdCount] = useState<number>(0);
  const [liveDetections, setLiveDetections] = useState<Detection[]>([]);
  const cameraRef = useRef<Camera>(null);

  const { model } = useTensorflowModel(
    Platform.select({
      ios: require('../../assets/models/yolov8n.mlpackage'), // Core ML on iOS
      android: require('../../assets/models/yolov8n_float32.tflite'), // TFLite on Android
    }),
    {
      delegate: Platform.select({ ios: 'core-ml', android: 'gpu' })
    }
  );
  const modelLoaded = model != null;

  // Ref to hold the loaded model for use in the worklet
  const modelRef = useRef<any | null>(null); // Using 'any' as 'Model' is not exported
  useEffect(() => {
    modelRef.current = model;
  }, [model]);

  const devices = useCameraDevices('back');
  const device = devices.back;

  const countScale = useSharedValue(1);
  const countOpacity = useSharedValue(1);

  const animatedCountStyle = useAnimatedStyle(() => {
    return {
      transform: [{ scale: countScale.value }],
      opacity: countOpacity.value,
    };
  });

  useEffect(() => {
    // Animate count change
    if (crowdCount > 0) {
      countScale.value = withSpring(1.2, {}, () => {
        countScale.value = withSpring(1);
      });
      countOpacity.value = withSpring(0.5, {}, () => {
        countOpacity.value = withSpring(1);
      });
    }
  }, [crowdCount]);

  useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'authorized');
      if (status !== 'authorized') {
        Alert.alert(
          "Camera Permission Required",
          "Please grant camera access to use this feature.",
          [{ text: "OK", onPress: () => navigation.goBack() }]
        );
      }
    })();
  }, []);

  const [isActive, setIsActive] = useState<boolean>(false);
  // useFocusEffect ensures the camera is active only when the screen is focused
  useFocusEffect(
    useCallback(() => {
      setIsActive(true);
      return () => setIsActive(false);
    }, [])
  );

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet'; // This function runs on the VisionCamera Worklet thread
    const currentModel = modelRef.current; // Get the current model from the ref

    if (!currentModel) {
      console.warn("AI Model not loaded for frame processor.");
      return;
    }

    // Process frame to TFLite input tensor (e.g., 640x640 RGB Float32)
    const inputTensor = processCameraFrame(frame);

    // Run TFLite inference using the model's 'run' method
    const output = currentModel.run([inputTensor]);
    const outputTensor = output[0] as Float32Array; // Output is typically Float32Array

    // Parse and filter YOLOv8 detections (only 'person' class)
    // Pass the actual frame dimensions for correct scaling of bounding boxes
    const detections = parseYOLOv8Output(outputTensor, frame.width, frame.height, 0.4);
    const personDetections = detections.filter(d => CLASS_LABELS[Math.round(d.classId)] === 'person');

    // Update React Native state on the JS thread
    runOnJS(setCrowdCount)(personDetections.length);
    runOnJS(setLiveDetections)(personDetections);
  }, []); // Dependencies for useFrameProcessor. Keep it empty if all dynamic values are accessed via refs.


  if (!hasPermission) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>Requesting Camera Permission...</Text>
      </View>
    );
  }

  if (device == null) {
    return (
      <View style={styles.permissionContainer}>
        <Text style={styles.permissionText}>Camera device not found. Check permissions or device.</Text>
      </View>
    );
  }

  // Calculate dimensions for camera preview and bounding box overlay
  // This aims to fit the camera preview to the width and maintain its aspect ratio.
  const cameraAspect = device.photoWidth / device.photoHeight;
  const previewWidth = width;
  const previewHeight = previewWidth / cameraAspect; // Maintain camera's aspect ratio

  // Scale detections from the camera frame's native resolution to the preview component's dimensions
  const overlayScaleX = previewWidth / device.photoWidth;
  const overlayScaleY = previewHeight / device.photoHeight;

  const scaledLiveDetections = liveDetections.map(det => {
    const [x1, y1, x2, y2] = det.box;
    return {
      ...det,
      box: [
        x1 * overlayScaleX,
        y1 * overlayScaleY,
        x2 * overlayScaleX,
        y2 * overlayScaleY,
      ]
    };
  });


  return (
    <View style={styles.liveContainer}>
      <Appbar.Header style={[styles.appbar, { backgroundColor: theme.colors.primary }]}>
        <Appbar.Action icon="arrow-left" onPress={() => navigation.goBack()} color="white" />
        <Appbar.Content title="Live Crowd Counter" titleStyle={styles.appbarTitle} />
        <View style={{ width: 48 }} /> {/* Placeholder for consistent spacing */}
      </Appbar.Header>

      <Camera
        ref={cameraRef}
        style={[styles.cameraPreview, { width: previewWidth, height: previewHeight }]}
        device={device}
        isActive={isActive}
        frameProcessor={frameProcessor}
        frameProcessorFps={10} // Process ~10 frames per second for real-time feel
        photo={true} // Enable photo capture if needed, even if not used directly for detection
      />

      {/* Overlay for drawing bounding boxes */}
      {liveDetections.length > 0 && (
        <BoundingBoxOverlay detections={scaledLiveDetections} imageWidth={previewWidth} imageHeight={previewHeight} type="live" />
      )}

      {/* Live count display */}
      <View style={styles.liveCountContainer}>
        <Animated.Text style={[styles.crowdCountText, animatedCountStyle, { color: 'white' }]}>
          Live Count: {crowdCount}
        </Animated.Text>
      </View>

      {/* Snackbar for model loading status */}
      {!modelLoaded && (
        <Snackbar
          visible={!modelLoaded}
          onDismiss={() => {}} // Indefinite duration
          duration={Snackbar.DURATION_INDETERMINATE}
          style={[styles.modelLoadingSnackbar, { backgroundColor: theme.colors.accent }]}
        >
          Loading AI Model... Please wait.
        </Snackbar>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  liveContainer: {
    flex: 1,
    backgroundColor: 'black', // Background when camera isn't filling
  },
  permissionContainer: {
    flex: 1,
    backgroundColor: 'black',
    justifyContent: 'center',
    alignItems: 'center',
  },
  appbar: {
    width: '100%',
    zIndex: 1, // Ensure app bar is on top
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    // Adjust padding for iOS safe areas (notches)
    paddingTop: Platform.OS === 'ios' ? 44 : 0, // Approx status bar height for iPhones
  },
  appbarTitle: {
    color: 'white',
    textAlign: 'center',
  },
  permissionText: {
    color: 'white',
    fontSize: 20,
    textAlign: 'center',
  },
  cameraPreview: {
    // Position absolutely to allow the BoundingBoxOverlay to sit on top of it.
    // The width and height are dynamically calculated to maintain aspect ratio.
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  liveCountContainer: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center', // Center horizontally
    backgroundColor: 'rgba(0,0,0,0.6)', // Semi-transparent black background
    paddingHorizontal: 25,
    paddingVertical: 15,
    borderRadius: 30,
    elevation: 5, // Android shadow
    shadowColor: '#000', // iOS shadow
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    zIndex: 2, // Ensure it's above camera and boxes
  },
  crowdCountText: {
    fontSize: 48,
    fontWeight: 'bold',
    color: 'white',
  },
  modelLoadingSnackbar: {
    position: 'absolute',
    bottom: 150, // Position above the count display
    left: 10,
    right: 10,
    zIndex: 3, // Ensure it's on top of everything
  }
});

export default CameraScreen;