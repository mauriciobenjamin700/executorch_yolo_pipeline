import 'dart:convert';
import 'dart:io';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import './pre_processing.dart';
import './results.dart';
import 'dart:math' as math;

Future<ExecuTorchModel> loadModel(String modelPath) async {
  final byteData = await rootBundle.load(modelPath);
  final tempDir = await getTemporaryDirectory();
  final modelTempName = modelPath.split('/').last;
  // Avoid appending an extra .pte if modelPath already contains the extension
  final file = File('${tempDir.path}/$modelTempName');
  await file.writeAsBytes(byteData.buffer.asUint8List());
  debugPrint('Loading model from: ${file.path}');
  try {
    final model = await ExecuTorchModel.load(file.path);
    debugPrint('ExecuTorchModel.load returned: $model');
    return model;
  } catch (e) {
    debugPrint('Erro ao carregar ExecuTorchModel: $e');
    rethrow;
  }
}

Future<List<String>> loadLabels(String labelsPath) async {
  final labelsData = await rootBundle.loadString(labelsPath);
  final labels = const LineSplitter().convert(labelsData);
  debugPrint('Rótulos carregados de $labelsPath');
  return labels;
}

class DetectionModel {
  final ExecuTorchModel model;
  final String modelPath;
  final int inputWidth;
  final int inputHeight;
  final List<String> labels;

  DetectionModel({
    required this.model,
    required this.modelPath,
    required this.inputWidth,
    required this.inputHeight,
    required this.labels,
  });

  Future<List<DetectionResult>> forward(
    Uint8List inputData, {
    double confThreshold = 0.25,
  }) async {
    final tensorData = await PreProcessing.toTensorData(
      inputData,
      targetWidth: inputWidth,
      targetHeight: inputHeight,
    );

    // Executa a inferência
    final outputs = await model.forward([tensorData]);

    debugPrint('Inferência concluída no modelo de detecção');
    for (var output in outputs) {
      debugPrint('Output shape: ${output.shape}');
      debugPrint('Output type: ${output.dataType}');
      debugPrint('Output data length: ${output.data.length}');
      debugPrint(
        'Output data (first 10 values): ${output.data.take(10).toList()}',
      );
    }
    if (outputs.isEmpty) return <DetectionResult>[];

    final out = outputs[0];
    // Expecting shape like [1, channels, num_boxes]
    final shape = out.shape;
    if (shape.length < 3) return <DetectionResult>[];
    final channels = shape[1]!;
    final numBoxes = shape[2]!;

    // Convert bytes to float32 list (respecting offset)
    final floatData = out.data.buffer.asFloat32List(
      out.data.offsetInBytes,
      out.data.lengthInBytes ~/ 4,
    );
    debugPrint('Parsed float length: ${floatData.length}');
    try {
      debugPrint('First 10 floats: ${floatData.take(10).toList()}');
    } catch (e) {
      debugPrint('Could not print float sample: $e');
    }

    // Helpers
    double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
    final List<DetectionResult> detections = [];

    final expectedWithObj = labels.length + 5;
    final hasObjectness = channels == expectedWithObj;
    final classCount = hasObjectness ? (channels - 5) : (channels - 4);
    debugPrint(
      'channels=$channels numBoxes=$numBoxes expectedWithObj=$expectedWithObj hasObjectness=$hasObjectness classCount=$classCount',
    );

    for (var i = 0; i < numBoxes; i++) {
      // index by channel-first layout: index = c * numBoxes + i
      double at(int c) => floatData[c * numBoxes + i];

      final x = at(0);
      final y = at(1);
      final w = at(2);
      final h = at(3);

      double objectness = 1.0;
      int classOffset = 4;
      if (hasObjectness) {
        objectness = sigmoid(at(4));
        classOffset = 5;
      }

      final classScores = List<double>.generate(
        classCount,
        (j) => at(classOffset + j),
      );
      if (i < 3)
        debugPrint(
          'box $i raw x,y,w,h: $x,$y,$w,$h objectness:$objectness classScoresSample:${classScores.take(6).toList()}',
        );
      final classProbs = classScores.map((s) => sigmoid(s)).toList();
      double maxClassProb = classProbs.reduce(math.max);
      final classId = classProbs.indexWhere((p) => p == maxClassProb);

      final conf = hasObjectness ? (objectness * maxClassProb) : maxClassProb;
      if (conf < confThreshold) continue;

      // Assume x,y,w,h are normalized center coords (0..1)
      final cx = x;
      final cy = y;
      final bw = w;
      final bh = h;

      double left = (cx - bw / 2.0) * inputWidth;
      double top = (cy - bh / 2.0) * inputHeight;
      double right = (cx + bw / 2.0) * inputWidth;
      double bottom = (cy + bh / 2.0) * inputHeight;

      // Clamp
      left = left.clamp(0.0, inputWidth.toDouble());
      top = top.clamp(0.0, inputHeight.toDouble());
      right = right.clamp(0.0, inputWidth.toDouble());
      bottom = bottom.clamp(0.0, inputHeight.toDouble());

      final label = (classId >= 0 && classId < labels.length)
          ? labels[classId]
          : 'class_$classId';

      detections.add(
        DetectionResult(
          classId: classId,
          label: label,
          confidence: conf,
          bbox: [left, top, right, bottom],
        ),
      );
    }

    // Sort by confidence descending
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    return detections;
  }


}


