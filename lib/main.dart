import 'dart:io';
import 'dart:typed_data';

import 'package:executorch_yolo_pipeline/core/pre_processing.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import './models/classify.dart';
import './models/detection.dart';
import './models/segment.dart';
import './core/results.dart';

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
  Uint8List? _imageBytes;
  Uint8List? _segmentedImageBytes;
  String _confidenceDetection = '—';
  String _classLabelDetection = '—';
  DetectionModel? _detectionModel;
  List<DetectionResult> _detections = [];
  ClassifyModel? _classificationModel;
  String _confidenceClassification = '—';
  String _classLabelClassification = '—'; 
  SegmentModel? _segmentationModel;


  Future<void> _loadModel() async {
    final detectionModel = await loadModel('assets/yolov8n.pte');
    final detectionLabels = await loadLabels('assets/detection_labels.txt');
    final classificationModel = await loadModel('assets/yolov8n-cls.pte');
    final classificationLabels = await loadLabels(
      'assets/classification_labels.txt',
    );
    final segmentationModel = await loadModel('assets/yolov8n-seg.pte');
    final segmentationLabels = await loadLabels(
      'assets/segmentation_labels.txt',
    );

    setState(() {
      _detectionModel = DetectionModel(
        model: detectionModel,
        modelPath: 'assets/yolov8n.pte',
        inputWidth: 640,
        inputHeight: 640,
        labels: detectionLabels,
      );
      _classificationModel = ClassifyModel(
        model: classificationModel,
        modelPath: 'assets/yolov8n-cls.pte',
        inputWidth: 224,
        inputHeight: 224,
        labels: classificationLabels,
      );
    });
    _segmentationModel = SegmentModel(
      model: segmentationModel,
      modelPath: 'assets/yolov8n-seg.pte',
      inputWidth: 640,
      inputHeight: 640,
      labels: segmentationLabels,
    );
  }

  Future<void> _pickImage() async {
    final XFile? picked = await _picker.pickImage(source: ImageSource.gallery);
    if (picked != null) {
      final bytes = await File(picked.path).readAsBytes();
      setState(() {
        _imageFile = File(picked.path);
        _imageBytes = bytes;
        _confidenceDetection = '—';
        _classLabelDetection = '—';
        _detections = [];
      });
    }
  }

  Future<void> _execute() async {
    if (_imageFile == null || _imageBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Selecione uma imagem primeiro')),
      );
      return;
    }
    if (_detectionModel == null) {
      await _loadModel();
    }

    try {
      final imageBytes = _imageBytes!;
      final detectionOutput = await _detectionModel!.predict(imageBytes);
      debugPrint('Detecção recebida: $detectionOutput');
      final classificationOutput = await _classificationModel!.predict(
        imageBytes,
      );
      // Guarda resultado de classificação
      setState(() {
        _confidenceClassification =
            '${(classificationOutput.confidence * 100).toStringAsFixed(1)}%';
        _classLabelClassification = classificationOutput.label;
      });
      debugPrint('\n\nClassificação recebida: $classificationOutput');
      try {
        final segmentationOutput = await _segmentationModel!.predict(
          imageBytes,
        );
        debugPrint('\n\nSegmentação recebida: $segmentationOutput');
        setState(() {
          _segmentedImageBytes = segmentationOutput.segmentedImage;
        });
      } catch (e) {
        // Caso a segmentação falhe, usa a imagem original como fallback
        debugPrint('Segmentação falhou: $e');
        setState(() {
          _segmentedImageBytes = imageBytes;
        });
      }

      setState(() {
        _detections = detectionOutput;
        if (detectionOutput.isNotEmpty) {
          final top = detectionOutput.first;
          _confidenceDetection =
              '${(top.confidence * 100).toStringAsFixed(1)}%';
          _classLabelDetection = top.label;
        } else {
          _confidenceDetection = 'Nenhuma detecção';
          _classLabelDetection = '—';
        }
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

            // Preview superior => agora com overlay de caixas
            if (_imageBytes != null) ...[
              Center(
                child: SizedBox(
                  width: 200,
                  height: 200,
                  child: Stack(
                    fit: StackFit.passthrough,
                    children: [
                      Image.memory(
                        _imageBytes!,
                        width: 200,
                        height: 200,
                        fit: BoxFit.cover,
                      ),
                      CustomPaint(
                        size: const Size(200, 200),
                        painter: BoundingBoxPainter(
                          _detections,
                          inputWidth: _detectionModel?.inputWidth ?? 640,
                          inputHeight: _detectionModel?.inputHeight ?? 640,
                        ),
                      ),
                    ],
                  ),
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

            // Área inferior com a mesma imagem e labels (mini preview com overlay)
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Row(
                  children: [
                    Container(
                      width: 120,
                      height: 120,
                      color: Colors.grey[100],
                      child: _imageBytes != null
                          ? Stack(
                              children: [
                                Image.memory(
                                  _imageBytes!,
                                  width: 120,
                                  height: 120,
                                  fit: BoxFit.cover,
                                ),
                                CustomPaint(
                                  size: const Size(120, 120),
                                  painter: BoundingBoxPainter(
                                    _detections,
                                    inputWidth:
                                        _detectionModel?.inputWidth ?? 640,
                                    inputHeight:
                                        _detectionModel?.inputHeight ?? 640,
                                  ),
                                ),
                              ],
                            )
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
                          Text(_confidenceDetection),
                          const SizedBox(height: 12),
                          const Text(
                            'Classe',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          const SizedBox(height: 6),
                          Text(_classLabelDetection),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),

            // Novos cards: Classificação e Detecção (imagem, confiança, classe)
            Column(
              children: [
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Row(
                      children: [
                        Container(
                          width: 100,
                          height: 100,
                          color: Colors.grey[100],
                          child: _imageBytes != null
                              ? Image.memory(
                                  _imageBytes!,
                                  width: 100,
                                  height: 100,
                                  fit: BoxFit.cover,
                                )
                              : const Center(child: Text('Imagem')),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Classificação',
                                style: TextStyle(fontWeight: FontWeight.bold),
                              ),
                              const SizedBox(height: 6),
                              Text(_classLabelClassification),
                              const SizedBox(height: 6),
                              Text(_confidenceClassification),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(12.0),
                    child: Row(
                      children: [
                        Container(
                          width: 100,
                          height: 100,
                          color: Colors.grey[100],
                          child: _segmentedImageBytes != null
                              ? Stack(
                                  children: [
                                    Image.memory(
                                      _segmentedImageBytes!,
                                      width: 100,
                                      height: 100,
                                      fit: BoxFit.cover,
                                    ),
                                    CustomPaint(
                                      size: const Size(100, 100),
                                      painter: BoundingBoxPainter(
                                        _detections,
                                        inputWidth:
                                            _detectionModel?.inputWidth ?? 640,
                                        inputHeight:
                                            _detectionModel?.inputHeight ?? 640,
                                      ),
                                    ),
                                  ],
                                )
                              : const Center(child: Text('Imagem')),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'Detecção',
                                style: TextStyle(fontWeight: FontWeight.bold),
                              ),
                              const SizedBox(height: 6),
                              Text(_classLabelDetection),
                              const SizedBox(height: 6),
                              Text(_confidenceDetection),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// Novo: painter para desenhar caixas e rótulos
class BoundingBoxPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final int inputWidth;
  final int inputHeight;

  BoundingBoxPainter(
    this.detections, {
    required this.inputWidth,
    required this.inputHeight,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..color = Colors.red;

    final fill = Paint()
      ..style = PaintingStyle.fill
      ..color = Colors.red.withValues(alpha: 0.6);

    for (final d in detections) {
      if (d.bbox.length < 4) continue;
      final left = d.bbox[0];
      final top = d.bbox[1];
      final right = d.bbox[2];
      final bottom = d.bbox[3];

      final sx = size.width / inputWidth;
      final sy = size.height / inputHeight;

      final rect = Rect.fromLTWH(
        left * sx,
        top * sy,
        (right - left) * sx,
        (bottom - top) * sy,
      );

      canvas.drawRect(rect, paint);

      final label = '${d.label} ${(d.confidence * 100).toStringAsFixed(1)}%';
      final textStyle = TextStyle(color: Colors.white, fontSize: 11);
      final tp = TextPainter(
        text: TextSpan(text: label, style: textStyle),
        textDirection: TextDirection.ltr,
      );
      tp.layout();

      final bgRect = Rect.fromLTWH(
        rect.left,
        rect.top - tp.height.clamp(12, 9999),
        tp.width + 6,
        tp.height,
      );
      canvas.drawRect(bgRect, fill);
      tp.paint(canvas, Offset(rect.left + 3, bgRect.top));
    }
  }

  @override
  bool shouldRepaint(covariant BoundingBoxPainter oldDelegate) {
    return oldDelegate.detections != detections ||
        oldDelegate.inputWidth != inputWidth ||
        oldDelegate.inputHeight != inputHeight;
  }
}
