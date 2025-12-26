import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

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

    // Placeholder: o usuário deve implementar a lógica real aqui.
    // Para integrar, substitua o conteúdo abaixo pela chamada ao seu modelo.
    setState(() {
      _confidence = 'Aguardando lógica';
      _classLabel = 'Aguardando lógica';
    });
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
