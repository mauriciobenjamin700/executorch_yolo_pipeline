/// Biblioteca de utilitários para operações relacionadas à segmentação de imagens.
/// Contém funções para processar detecções, construir máscaras binárias e aplicar máscaras a imagens.
/// 
/// Functions:
/// - getBestSegmentationIndex: Retorna o índice da melhor detecção de segmentação com base em um limiar de confiança.
/// - extractMaskCoefficients: Extrai os coeficientes da máscara para uma detecção específica.
/// - buildBinaryMask: Constrói uma máscara binária a partir dos protótipos de máscara e coeficientes.
/// - decodeOriginalImage: Decodifica uma imagem original a partir de bytes Uint8List.
/// - resizeMask: Redimensiona a máscara para as dimensões alvo.
/// - applyMaskToImage: Aplica a máscara binária à imagem original.
/// - encodeImageToPng: Codifica uma imagem em bytes PNG (Uint8List
/// - getMaxIndexAndProb: Obtém o índice e a probabilidade máxima de uma lista de probabilidades.
library;

import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// Retorna o índice da melhor detecção de segmentação com base em um limiar de confiança.  
/// 
/// Params:
///   - detections: Lista 2D onde cada coluna representa uma detecção e cada linha um atributo (incluindo confiança).
///   - threshold: Limiar mínimo de confiança para considerar uma detecção válida.
/// 
/// Returns: Índice da melhor detecção ou -1 se nenhuma detecção atender ao limiar.
int getBestSegmentationIndex(List<List<double>> detections, double threshold) {
  double maxConfidence = -1.0;
  int bestDetectionIndex = -1;
  int detectionCount = detections[0].length;

  for (int i = 0; i < detectionCount; i++) {
    final confidence = detections[4][i];
    if (confidence > 0.5 && confidence > maxConfidence) {
      maxConfidence = confidence;
      bestDetectionIndex = i;
    }
  }

  return bestDetectionIndex;
}

/// Extrai os coeficientes da máscara para uma detecção específica.
///
/// Params:
///   - detections: Lista 2D onde cada coluna representa uma detecção e cada  linha um atributo (incluindo coeficientes da máscara).
///   - detectionIndex: Índice da detecção da qual extrair os coeficientes.
///   - firstCoeffIndex: Índice da primeira coluna que contém os coeficientes da máscara.
/// 
/// Returns: Lista de coeficientes da máscara.
List<double> extractMaskCoefficients(
  List<List<double>> detections,
  int detectionIndex,
  int firstCoeffIndex,
) {
  final maskCoeffs = <double>[];
  for (int i = firstCoeffIndex; i < detections.length; i++) {
    maskCoeffs.add(detections[i][detectionIndex]);
  }
  return maskCoeffs;
}

/// Constrói uma máscara binária a partir dos protótipos de máscara e coeficientes.
///
/// Params:
///   - maskPrototypes: Lista 3D representando os protótipos de máscara (altura x largura x canais).
///   - maskCoefficients: Lista de coeficientes da máscara. 
/// 
/// Returns: Imagem binária resultante.
img.Image buildBinaryMask(
  List<List<List<double>>> maskPrototypes,
  List<double> maskCoefficients,
) {
  final height = maskPrototypes.length;
  final width = maskPrototypes[0].length;
  final channels = maskPrototypes[0][0].length;
  final numCoeffs = math.min(maskCoefficients.length, channels);

  final binaryMask = img.Image(width: width, height: height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double maskValue = 0.0;

      // eixo dos canais no final (y, x, i)
      for (int i = 0; i < numCoeffs; i++) {
        maskValue += maskCoefficients[i] * maskPrototypes[y][x][i];
      }

      // Aplicar função sigmoide
      final sigmoidValue = 1 / (1 + math.exp(-maskValue));

      // Binarizar com threshold 0.5
      final pixelValue = sigmoidValue > 0.5 ? 255 : 0;

      binaryMask.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
    }
  }

  return binaryMask;
}

/// Decodifica uma imagem original a partir de bytes Uint8List.
///
/// Params:
///   - bytes: Bytes da imagem original.
/// 
/// Returns: Imagem decodificada.
img.Image decodeOriginalImage(Uint8List bytes) {
  final image = img.decodeImage(bytes);
  if (image == null) {
    throw Exception("Falha ao decodificar a imagem original.");
  }
  return image;
}

/// Redimensiona a máscara para as dimensões alvo.
///
/// Params:
///   - mask: Imagem da máscara a ser redimensionada.
///  - targetWidth: Largura alvo.
/// - targetHeight: Altura alvo.
/// 
/// Returns: Imagem da máscara redimensionada.
img.Image resizeMask(img.Image mask, int targetWidth, int targetHeight) {
  return img.copyResize(
    mask,
    width: targetWidth,
    height: targetHeight,
    interpolation: img.Interpolation.linear,
  );
}

/// Aplica a máscara binária à imagem original.
/// Zera os pixels da imagem original onde a máscara é preta.
/// 
/// Params:
///  - image: Imagem original.
///  - mask: Máscara binária.
/// 
/// Returns: Imagem resultante com a máscara aplicada.
img.Image applyMaskToImage(img.Image image, img.Image mask) {
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final maskPixel = mask.getPixel(x, y);
      // Canal R, G e B são iguais, então usamos R
      if (maskPixel.r < 128) {
        image.setPixelRgb(x, y, 0, 0, 0);
      }
    }
  }
  return image;
}

/// Codifica uma imagem em bytes PNG (Uint8List).
/// 
/// Params:
///   - image: Imagem a ser codificada.
/// 
/// Returns: Bytes PNG da imagem.
Uint8List encodeImageToPng(img.Image image) {
  return Uint8List.fromList(img.encodePng(image));
}


/// Obtém o índice e a probabilidade máxima de uma lista de probabilidades.
/// 
/// Params:
///   - probabilities: Lista de probabilidades para cada classe.
/// 
/// Returns: Tupla contendo o índice da classe com a maior probabilidade e o valor dessa probabilidade.
(int, double) getMaxIndexAndProb(List<double> probabilities) {
  double maxProb = 0.0;
  int maxIndex = -1;
  for (int i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }
  if (maxIndex == -1) {
    throw Exception('No classification result found.');
  }

  return (maxIndex, maxProb);
}
