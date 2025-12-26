import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:executorch_flutter/executorch_flutter.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Executorch YOLO Pipeline',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color.fromARGB(255, 183, 87, 58),
        ),
      ),
      home: const MyHomePage(title: 'Executorch YOLO Pipeline'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final ImagePicker _picker = ImagePicker();
  File? _imageFile;
  String _confidence = '—';
  String _classLabel = '—';
  ExecuTorchModel? _model;

  Future<void> _loadModel() async {
    final byteData = await rootBundle.load('assets/yolov8n.pte');
    final tempDir = await getTemporaryDirectory();
    final file = File('${tempDir.path}/model.pte');
    await file.writeAsBytes(byteData.buffer.asUint8List());
    final aux = await ExecuTorchModel.load(file.path);
    setState(() {
      _model = aux;
    });
  }

  // Preprocessar imagem para o formato esperado pelo modelo
  Future<Uint8List> _preprocessImage(File imageFile) async {
    final imageBytes = await imageFile.readAsBytes();
    final image = img.decodeImage(imageBytes);

    if (image == null) {
      throw Exception('Falha ao decodificar imagem');
    }

    // Redimensionar para 640x640 (tamanho esperado pelo YOLO11n)
    final resized = img.copyResize(image, width: 640, height: 640);
    final normalizedData = <double>[];
    final mean = [0.485, 0.456, 0.406];
    final std = [0.229, 0.224, 0.225];
    
    for (int i = 0; i < resized.height; i++) {
      for (int j = 0; j < resized.width; j++) {
        final pixel = resized.getPixelSafe(j, i);

        // Extrair RGB (valores de 0 a 255)
        final r = pixel.rNormalized.toDouble();
        final g = pixel.gNormalized.toDouble();
        final b = pixel.bNormalized.toDouble();

        // Normalizar (ImageNet)
        normalizedData.add((r - mean[0]) / std[0]); // R
        normalizedData.add((g - mean[1]) / std[1]); // G
        normalizedData.add((b - mean[2]) / std[2]); // B
      }
    }

    // Converter para Float32List
    final float32Data = Float32List.fromList(normalizedData);
    return float32Data.buffer.asUint8List();
  }

  Future<void> _pickImage() async {
    final XFile? picked = await _picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      setState(() {
        _imageFile = File(picked.path);
        _confidence = '—';
        _classLabel = '—';
      });
    }
  }

  Future<void> _execute() async {
    if (_imageFile == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Selecione uma imagem primeiro')),
      );
      return;
    }
    if (_model == null) {
      await _loadModel();
    }

    try {
      // Preprocessar a imagem
      final processedImageData = await _preprocessImage(_imageFile!);

      final inputTensor = TensorData(
        shape: [1, 3, 640, 640],
        dataType: TensorType.float32,
        data: processedImageData,
      );

      final outputs = await _model!.forward([inputTensor]);

      for (var output in outputs) {
        debugPrint('Output shape: ${output.shape}');
        debugPrint('Output type: ${output.dataType}');
        debugPrint('Output data length: ${output.data.length}');
        debugPrint(
          'Output data (first 10 values): ${output.data.take(10).toList()}',
        );
      }

      // TODO: Implementar lógica para extrair confiança e classe dos outputs
      setState(() {
        _confidence = 'Aguardando lógica';
        _classLabel = 'Aguardando lógica';
      });
    } catch (e) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Erro: $e')));
      debugPrint('Erro na execução: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _pickImage,
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Selecionar da galeria'),
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton.icon(
                  onPressed: _execute,
                  icon: const Icon(Icons.play_arrow),
                  label: const Text('Executar'),
                ),
              ],
            ),

            const SizedBox(height: 20),

            // Preview superior
            if (_imageFile != null) ...[
              Center(
                child: Image.file(
                  _imageFile!,
                  width: 200,
                  height: 200,
                  fit: BoxFit.cover,
                ),
              ),
              const SizedBox(height: 20),
            ] else ...[
              Container(
                height: 200,
                color: Colors.grey[200],
                child: const Center(child: Text('Nenhuma imagem selecionada')),
              ),
              const SizedBox(height: 20),
            ],

            // Área inferior com a mesma imagem e labels
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Row(
                  children: [
                    Container(
                      width: 120,
                      height: 120,
                      color: Colors.grey[100],
                      child: _imageFile != null
                          ? Image.file(_imageFile!, fit: BoxFit.cover)
                          : const Center(child: Text('Imagem')),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Confiança',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 6),
                          Text(_confidence),
                          const SizedBox(height: 12),
                          const Text(
                            'Classe',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 6),
                          Text(_classLabel),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
