// src/screens/HomeScreen.tsx
import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, Image, ActivityIndicator, Dimensions, Platform, Alert } from 'react-native';
import { Button, Appbar, Snackbar, useTheme } from 'react-native-paper';
import { launchImageLibrary } from 'react-native-image-picker';
import { useTensorflowModel } from 'react-native-fast-tflite';
import Animated, { useSharedValue, useAnimatedStyle, withSpring } from 'react-native-reanimated';
import { useNavigation } from '@react-navigation/native';
import BoundingBoxOverlay from '../components/BoundingBoxOverlay';
import { Detection } from '../utils/modelUtils'; // Import Detection type
import { preprocessImageForTFLite } from '../utils/imageProcessing'; // Placeholder for static image preprocessing

const { width, height } = Dimensions.get('window');

const HomeScreen: React.FC = () => {
  const navigation = useNavigation();
  const theme = useTheme();

  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [crowdCount, setCrowdCount] = useState<number>(0);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const { model } = useTensorflowModel(
    Platform.select({
      ios: require('../../assets/models/yolov8n.mlpackage'), // Core ML on iOS
      android: require('../../assets/models/yolov8n_float32.tflite'), // TFLite on Android
    }),
    {
        delegate: Platform.select({
            ios: { type: 'core-ml' },
            android: { type: 'gpu' }
        })
    }
  );
  const modelLoaded = model != null;

  const countScale = useSharedValue(1);
  const countOpacity = useSharedValue(1);

  const animatedCountStyle = useAnimatedStyle(() => {
    return {
      transform: [{ scale: countScale.value }],
      opacity: countOpacity.value,
    };
  });

  useEffect(() => {
    if (crowdCount > 0) {
      countScale.value = withSpring(1.2, {}, () => {
        countScale.value = withSpring(1);
      });
      countOpacity.value = withSpring(0.5, {}, () => {
        countOpacity.value = withSpring(1);
      });
    }
  }, [crowdCount]);

  const handleImagePick = async () => {
    setLoading(true);
    setSelectedImage(null);
    setCrowdCount(0);
    setDetections([]);
    setError(null);

    if (!modelLoaded) {
      setError("AI Model not loaded. Please wait.");
      setLoading(false);
      return;
    }

    try {
      const result = await launchImageLibrary({ mediaType: 'photo', quality: 0.8, includeExtra: true });

      if (result.didCancel) {
        console.log('User cancelled image picker');
        return;
      }
      if (result.errorCode) {
        console.error('Image picker error:', result.errorMessage);
        setError(`Image picker error: ${result.errorMessage}`);
        return;
      }

      if (result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        setSelectedImage(asset.uri || null);

        // --- IMPORTANT: This is a placeholder for static image processing ---
        // For a real app, `preprocessImageForTFLite` needs to load the image URI,
        // resize it to 640x640, convert to RGB Float32Array, and normalize.
        // This is complex and often requires a native module.
        // For now, we'll simulate the inference result.
        // const inputBuffer = await preprocessImageForTFLite(asset.uri!);
        // const { detections: newDetections, count: newCount } = await performInference(model, inputBuffer, asset.width!, asset.height!);

        const { detections: newDetections, count: newCount } = simulateYOLOv8Inference(asset.width || width, asset.height || height);

        setDetections(newDetections);
        setCrowdCount(newCount);
      }
    } catch (err: any) {
      console.error('Error during image processing:', err);
      setError(`Failed to process image: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const navigateToCamera = () => {
    if (!modelLoaded) {
      setError("AI Model not loaded. Please wait.");
      return;
    }
    navigation.navigate('CameraScreen' as never); // Type assertion for navigation
  };

  return (
    <View style={styles.container}>
      <Appbar.Header style={[styles.appbar, { backgroundColor: theme.colors.primary }]}>
        <Appbar.Content title="Crowd Counter" titleStyle={styles.appbarTitle} />
      </Appbar.Header>

      <View style={styles.imagePreviewContainer}>
        {selectedImage ? (
          <>
            <Image source={{ uri: selectedImage }} style={styles.imagePreview} />
            {detections.length > 0 && (
              <BoundingBoxOverlay detections={detections} imageWidth={width * 0.9} imageHeight={height * 0.6} type="static" />
            )}
          </>
        ) : (
          <Text style={styles.placeholderText}>Upload an image or take a photo</Text>
        )}
        {loading && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color={theme.colors.primary} />
            <Text style={styles.loadingText}>Analyzing...</Text>
          </View>
        )}
      </View>

      <Animated.Text style={[styles.crowdCountText, animatedCountStyle, { color: theme.colors.primary }]}>
        Crowd Count: {crowdCount}
      </Animated.Text>

      <View style={styles.buttonContainer}>
        <Button
          mode="contained"
          icon="image-plus"
          onPress={handleImagePick}
          style={[styles.actionButton, { backgroundColor: theme.colors.primary }]}
          labelStyle={styles.actionButtonLabel}
          loading={loading}
          disabled={loading || !modelLoaded}
        >
          Upload Image
        </Button>
        <Button
          mode="contained"
          icon="camera"
          onPress={navigateToCamera}
          style={[styles.actionButton, { backgroundColor: theme.colors.primary }]}
          labelStyle={styles.actionButtonLabel}
          loading={loading}
          disabled={loading || !modelLoaded}
        >
          Live Camera
        </Button>
      </View>

      <Snackbar
        visible={!!error}
        onDismiss={() => setError(null)}
        action={{
          label: 'Dismiss',
          onPress: () => setError(null),
        }}
        style={[styles.snackbar, { backgroundColor: theme.colors.error }]}
      >
        {error}
      </Snackbar>

      {!modelLoaded && (
        <Snackbar
          visible={!modelLoaded}
          onDismiss={() => {}}
          duration={Snackbar.DURATION_INDETERMINATE}
          style={[styles.modelLoadingSnackbar, { backgroundColor: theme.colors.accent }]}
        >
          Loading AI Model... Please wait.
        </Snackbar>
      )}
    </View>
  );
};

// --- SIMULATED INFERENCE FOR DEMO PURPOSES ---
// This will provide fake detections until `preprocessImageForTFLite` is properly implemented.
const simulateYOLOv8Inference = (imgWidth: number, imgHeight: number): { detections: Detection[]; count: number } => {
  const detections: Detection[] = [];
  const numPeople = Math.floor(Math.random() * 20) + 5; // Simulate 5-25 people
  for (let i = 0; i < numPeople; i++) {
    const x = Math.random() * imgWidth * 0.8;
    const y = Math.random() * imgHeight * 0.8;
    const w = Math.random() * (imgWidth * 0.2) + 20;
    const h = Math.random() * (imgHeight * 0.2) + 20;
    detections.push({
      box: [x, y, x + w, y + h], // [x1, y1, x2, y2]
      confidence: Math.random() * 0.3 + 0.6, // Simulate confidence between 60-90%
      classId: 0, // 'person'
    });
  }
  return { detections, count: detections.length };
};
// --- END SIMULATED INFERENCE ---

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
    alignItems: 'center',
    paddingTop: Platform.OS === 'android' ? Appbar.HEIGHT : 0,
  },
  appbar: {
    width: '100%',
  },
  appbarTitle: {
    color: 'white',
    textAlign: 'center',
  },
  imagePreviewContainer: {
    width: width * 0.9,
    height: height * 0.6,
    backgroundColor: '#E0E0E0',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 15,
    overflow: 'hidden',
    marginTop: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: '#CCC',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  imagePreview: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  placeholderText: {
    color: '#888',
    fontSize: 18,
    textAlign: 'center',
    padding: 20,
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 10,
  },
  loadingText: {
    marginTop: 10,
    fontSize: 16,
    color: '#007BFF',
    fontWeight: 'bold',
  },
  crowdCountText: {
    fontSize: 60,
    fontWeight: 'bold',
    marginVertical: 20,
    textAlign: 'center',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '95%',
    position: 'absolute',
    bottom: 30,
  },
  actionButton: {
    marginHorizontal: 10,
    paddingVertical: 12,
    paddingHorizontal: 25,
    borderRadius: 30,
    elevation: 3,
  },
  actionButtonLabel: {
    fontSize: 18,
    color: 'white',
    fontWeight: 'bold',
  },
  snackbar: {
    position: 'absolute',
    bottom: 80,
    left: 10,
    right: 10,
  },
  modelLoadingSnackbar: {
    position: 'absolute',
    bottom: 80,
    left: 10,
    right: 10,
  }
});

export default HomeScreen;