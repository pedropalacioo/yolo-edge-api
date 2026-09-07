# yolo-edge-api

Projeto de visão computacional para Edge AI desenvolvido ao longo das aulas. O
repositório reúne a API de inferência, streaming, pré-processamento, testes,
versionamento de modelos e datasets com DVC e os experimentos executados.

## Aula 6 — Transfer Learning com MobileNetV2

Na Aula 6, o dataset `epi-v1` foi adaptado de detecção para classificação por
meio de recortes das caixas anotadas. A MobileNetV2 pré-treinada no ImageNet foi
executada no Google Colab com GPU NVIDIA Tesla T4 e CUDA durante 10 épocas.

- [Notebook executado](notebooks/aula6/notebook_mobilenet_transfer_learning_aula6_executado.ipynb)
- [Resultados, método e reprodução](docs/aula6/README.md)
- [Notebook no Google Colab](https://colab.research.google.com/drive/1szYYHF-szyJLAlXPfVDJLIH4hA0InqWn?usp=sharing)

Resultado final de `model.evaluate(val_ds)`: **92,98% de acurácia**, com loss
de **0,184989** sobre 114 recortes de validação.
