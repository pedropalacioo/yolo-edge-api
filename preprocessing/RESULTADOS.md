# Resultados dos experimentos — Aula 5

Modelo: `models/yolo-epi.pt` (YOLOv8n, 3 classes, 30 épocas). Dataset:
`epi-v1`, split de validação com 20 imagens e 114 instâncias. Todas as
avaliações foram executadas com `imgsz=640`. Baseline independente: `0,7562`.

| Experimento | Configuração | mAP@0.5 (val) | Delta vs baseline | Pré-proc. médio (ms) |
|---|---|---:|---:|---:|
| E1-A | BGR sem conversão | 0,6613 | -0,0949 | 0,378 |
| E1-B | RGB correto | 0,7572 | +0,0010 | 0,370 |
| E2-A | Resize simples | 0,7572 | +0,0010 | 0,327 |
| E2-B | Letterbox correto | 0,7562 | +0,0000 | 0,000¹ |
| E3-A | Sem filtro | 0,7572 | +0,0010 | 0,382 |
| E3-B | GaussianBlur 3x3, sigma=0,8 | 0,7523 | -0,0039 | 2,271 |
| E3-C | GaussianBlur 5x5, sigma=1,5 | 0,7112 | -0,0451 | 2,936 |
| E3-D | medianBlur kernel=3 | 0,7789 | +0,0227 | 1,400 |
| E4-A | Sem equalização, baixa luz | 0,7647 | +0,0084 | 0,992 |
| E4-B | equalizeHist global | 0,7676 | +0,0114 | 15,961 |
| E4-C | CLAHE clip=2, tile=8 | 0,7585 | +0,0022 | 12,132 |

¹ E2-B usa o letterbox interno validado da Ultralytics; seu custo aparece na
métrica de preprocessamento do framework e não no transformador externo.

Os deltas muito pequenos (cerca de 0,001) são tratados como equivalentes ao
baseline. Neste conjunto, o filtro mediano foi o melhor filtro normal. Para
baixa luz sintética, `equalizeHist` obteve o maior mAP, mas CLAHE permanece no
preset `CONFIG_LOW_LIGHT` por preservar contraste local e custar menos nesta
medição; a escolha deve ser reavaliada com imagens reais da câmera.
