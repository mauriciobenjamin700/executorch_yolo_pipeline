import 'dart:convert';
import 'dart:io';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import './pre_processing.dart';

class PytorchModel {
  ExecuTorchModel? model;
  String modelPath;
  List<String> labels;
  int inputWidth;
  int inputHeight;

  PytorchModel({
    required ExecuTorchModel model,
    required this.modelPath,
    required this.inputWidth,
    required this.inputHeight,
    required this.labels,
  });

  static Future<PytorchModel> loadModel(
    String modelPath,
    String labelsPath,
    int inputWidth,
    int inputHeight,
  ) async {
    // Carrega o modelo do arquivo de ativos para um diretório temporário
    final byteData = await rootBundle.load(modelPath);
    final tempDir = await getTemporaryDirectory();
    final modelTempName = modelPath.split('/').last;
    final file = File('${tempDir.path}/$modelTempName.pte');
    await file.writeAsBytes(byteData.buffer.asUint8List());
    final aux = await ExecuTorchModel.load(file.path);

    // Carrega os rótulos das classes
    final labelsData = await rootBundle.loadString(labelsPath);
    final labels = const LineSplitter().convert(labelsData);
    debugPrint('Rótulos carregados de $labelsPath');

    // Retorna a instância do modelo
    debugPrint('Modelo carregado de $modelPath');
    return PytorchModel(
      model: aux,
      modelPath: modelPath,
      inputWidth: inputWidth,
      inputHeight: inputHeight,
      labels: labels,
    );
  }

  Future<void> dispose() async {
    await model?.dispose();
    debugPrint('Modelo descarregado de $modelPath');
  }
}

class DetectionModel extends PytorchModel {
  DetectionModel({
    required super.model,
    required super.modelPath,
    required super.inputWidth,
    required super.inputHeight,
    required super.labels,
  });

  Future<TensorData> forward(Uint8List inputData) async {
    if (model == null) {
      throw Exception('Modelo não carregado');
    }

    final tensorData = await PreProcessing.toTensorData(
      inputData,
      targetWidth: inputWidth,
      targetHeight: inputHeight,
    );

    // Executa a inferência
    final outputs = await model!.forward([tensorData]);

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
