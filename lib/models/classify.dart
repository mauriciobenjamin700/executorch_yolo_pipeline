

import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:executorch_yolo_pipeline/core/results.dart';
import 'package:executorch_yolo_pipeline/core/utils.dart';

class ClassifyModel {
  
  final ExecuTorchModel model;
  final String modelPath;
  final int inputWidth;
  final int inputHeight;
  final List<String> labels;

  ClassifyModel({
    required this.model,
    required this.modelPath,
    required this.inputWidth,
    required this.inputHeight,
    required this.labels,
  });


  /// Realiza a predição de classificação em uma imagem de entrada.
  /// Retorna um [ClassificationResult] contendo os resultados da classificação.
  Future<ClassificationResult> predict(
    TensorData inputTensor,
    Uint8List originalImageBytes,
  ) async {
    final outputs = await forward(inputTensor, originalImageBytes);
    return getResult(outputs, originalImageBytes);
  }

  /// Executa a inferência do modelo com o tensor de entrada.
  /// Retorna um mapa contendo as saídas do modelo.
  Future<List<TensorData>> forward(
    TensorData inputTensor,
    Uint8List originalImageBytes,
  ) async {
    try {
      final output = await model.forward([inputTensor]);
      return output;

    } catch (error) {
      rethrow;
    }
  }

  /// Processa as saídas do modelo para gerar o resultado de classificação
  /// Retorna um [ClassificationResult] contendo os resultados da classificação.
  ClassificationResult getResult (
    List<TensorData> outputs,
    Uint8List originalImageBytes,
  ) {
    final probabilities = (outputs[0] as List<List<double>>)[0];
    final (maxIndex, maxProb) = getMaxIndexAndProb(probabilities);
    
    return ClassificationResult(
      originalImage: originalImageBytes,
      label: labels[maxIndex],
      confidence: maxProb,
      allProbabilities: probabilities,
      timestamp: DateTime.now(),
    );
  }
}