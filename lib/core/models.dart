import 'dart:convert';
import 'dart:io';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import './pre_processing.dart';

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


  Future<TensorData> forward(Uint8List inputData) async {
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

    return outputs[0];
  }
}
