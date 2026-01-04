import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_yolo_pipeline/core/results.dart';
import 'package:executorch_yolo_pipeline/core/utils.dart';

class SegmentModel {
  final ExecuTorchModel model;
  final String modelPath;
  final int inputWidth;
  final int inputHeight;
  final List<String> labels;

  SegmentModel({
    required this.model,
    required this.modelPath,
    required this.inputWidth,
    required this.inputHeight,
    required this.labels,
  });

  Future<SegmentationResult> predict(
    TensorData inputTensor,
    Uint8List originalImageBytes,
  ) async {
    final outputs = await forward(inputTensor, originalImageBytes);
    return getResult(outputs, originalImageBytes);
  }

  Future<List<TensorData>> forward(
    TensorData inputTensor,
    Uint8List originalImageBytes,
  ) async {
    try {
      final outputs = await model.forward([inputTensor]);

      return outputs;
    } catch (error) {
      rethrow;
    }
  }

  SegmentationResult getResult(
    List<TensorData> outputs,
    Uint8List originalImageBytes,
  ) {
    final segmentationThreshold = 0.5; // Threshold de confiança
    final firstCoeffIndex =
        5; // Índice da primeira coluna dos coeficientes da máscara
    double confidence;

    final segmentations = (outputs[0].data as List<List<List<double>>>)[0];
    final maskPrototypes = (outputs[1].data as List<List<List<List<double>>>>)[0];

    final bestSegmentationIndex = getBestSegmentationIndex(
      segmentations,
      segmentationThreshold,
    );
    if (bestSegmentationIndex == -1) {
      throw Exception("No found segmentations.");
    }

    final maskCoeffs = extractMaskCoefficients(
      segmentations,
      bestSegmentationIndex,
      firstCoeffIndex,
    );

    final binaryMask = buildBinaryMask(maskPrototypes, maskCoeffs);
    final originalImage = decodeOriginalImage(originalImageBytes);
    final resizedMask = resizeMask(
      binaryMask,
      originalImage.width,
      originalImage.height,
    );
    final maskedImage = applyMaskToImage(originalImage, resizedMask);
    final segmentedImageBytes = encodeImageToPng(maskedImage);

    confidence = bestSegmentationIndex < segmentations[0].length
        ? segmentations[4][bestSegmentationIndex]
        : 0.0;

    return SegmentationResult(
      originalImage: originalImageBytes,
      segmentedImage: segmentedImageBytes,
      binaryMask: binaryMask,
      maskCoefficients: maskCoeffs,
      bestSegmentationIndex: bestSegmentationIndex,
      confidence: confidence,
      timestamp: DateTime.now(),
      metadata: {
        'threshold': segmentationThreshold,
        'firstCoeffIndex': firstCoeffIndex,
        'originalImageSize': '${originalImage.width}x${originalImage.height}',
      },
    );
  }
}
