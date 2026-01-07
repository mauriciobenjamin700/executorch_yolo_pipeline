import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_yolo_pipeline/core/pre_processing.dart';
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

  Future<SegmentationResult> predict(Uint8List originalImageBytes) async {
    final inputTensor = await PreProcessing.toTensorData(
      originalImageBytes,
      targetWidth: inputWidth,
      targetHeight: inputHeight,
    );
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
    // Converte outputs[0] -> segmentations (esperado como [1, channels, num_detections])
    final outSeg = outputs[0];
    // Ler floats de forma segura mesmo que o Uint8List seja um view desalinhado
    final bdSeg = ByteData.sublistView(outSeg.data);
    final floatLenSeg = bdSeg.lengthInBytes ~/ 4;
    final floatSeg = Float32List(floatLenSeg);
    for (var i = 0; i < floatLenSeg; i++) {
      floatSeg[i] = bdSeg.getFloat32(i * 4, Endian.little);
    }
    final shapeSeg = outSeg.shape;
    final channels = shapeSeg.length >= 3 ? shapeSeg[1]! : 0;
    final numDet = shapeSeg.length >= 3 ? shapeSeg[2]! : 0;

    final segmentations = List<List<double>>.generate(
      channels,
      (_) => List<double>.filled(numDet, 0.0),
    );
    for (int c = 0; c < channels; c++) {
      for (int i = 0; i < numDet; i++) {
        segmentations[c][i] = floatSeg[c * numDet + i];
      }
    }

    // Converte outputs[1] -> mask prototypes (pode estar em NCHW ou NHWC)
    final outProto = outputs[1];
    final bdProto = ByteData.sublistView(outProto.data);
    final floatLenProto = bdProto.lengthInBytes ~/ 4;
    final floatProto = Float32List(floatLenProto);
    for (var i = 0; i < floatLenProto; i++) {
      floatProto[i] = bdProto.getFloat32(i * 4, Endian.little);
    }
    final shapeProto = outProto.shape;

    late final List<List<List<double>>> maskPrototypes;
    if (shapeProto.length >= 4) {
      final a = shapeProto[1]!;
      final b = shapeProto[2]!;
      final d = shapeProto[3]!;
      // Detecta se o layout é NCHW ([1, C, H, W]) ou NHWC ([1, H, W, C])
      final floatLen = floatProto.length;
      final maybeC = floatLen ~/ (b * d);
      if (maybeC == a) {
        // NCHW -> converter para [H][W][C]
        final C = a;
        final H = b;
        final W = d;
        maskPrototypes = List.generate(
          H,
          (_) => List.generate(W, (_) => List<double>.filled(C, 0.0)),
        );
        for (int c = 0; c < C; c++) {
          final base = c * H * W;
          for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
              maskPrototypes[y][x][c] = floatProto[base + y * W + x];
            }
          }
        }
      } else {
        // NHWC -> [1, H, W, C]
        final H = a;
        final W = b;
        final C = d;
        maskPrototypes = List.generate(
          H,
          (_) => List.generate(W, (_) => List<double>.filled(C, 0.0)),
        );
        for (int y = 0; y < H; y++) {
          for (int x = 0; x < W; x++) {
            final base = (y * W + x) * C;
            for (int c = 0; c < C; c++) {
              maskPrototypes[y][x][c] = floatProto[base + c];
            }
          }
        }
      }
    } else {
      throw Exception('Formato inesperado para protótipos de máscara');
    }

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

    final segLabel =
        (bestSegmentationIndex >= 0 && bestSegmentationIndex < labels.length)
        ? labels[bestSegmentationIndex]
        : null;

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
        'label': segLabel,
      },
    );
  }
}
