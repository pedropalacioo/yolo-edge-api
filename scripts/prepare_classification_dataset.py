"""Cria recortes de classificação sem modificar o epi-v1. Saída deve ser nova."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image, ImageDraw, ImageOps

CLASSES = ["Capacete", "Colete", "Pessoa"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(
            "A saída já existe; escolha uma pasta nova para não sobrescrever resultados."
        )
    root = args.output / "datasets/aula6_classificacao"
    ev = args.output / "evidencias_aula6"
    ev.mkdir(parents=True)
    for split in ["train", "validation"]:
        for name in CLASSES:
            (root / split / name).mkdir(parents=True)
    rows, rejected, seen = [], [], {}
    source_hashes = {}
    for original_split, split in [("train", "train"), ("valid", "validation")]:
        images = sorted((args.source / original_split / "images").iterdir())
        labels = args.source / original_split / "labels"
        assert {p.stem for p in images} == {p.stem for p in labels.glob("*.txt")}, (
            "Imagem/label sem par"
        )
        for path in images:
            label = labels / (path.stem + ".txt")
            source_hashes[str(path.relative_to(args.source))] = sha(path)
            source_hashes[str(label.relative_to(args.source))] = sha(label)
            with Image.open(path) as im:
                im.load()
                if im.getexif().get(274, 1) != 1:
                    raise ValueError(f"Orientação EXIF exige revisão: {path}")
                im = im.convert("RGB")
                w, h = im.size
                for line_no, line in enumerate(label.read_text().splitlines(), 1):
                    if not line.strip():
                        continue
                    fields = line.split()
                    if len(fields) != 5:
                        raise ValueError(
                            f"Label fora do formato YOLO: {label}:{line_no}"
                        )
                    cls, cx, cy, bw, bh = map(float, fields)
                    if not all(math.isfinite(x) for x in [cls, cx, cy, bw, bh]):
                        raise ValueError(f"Label não finito: {label}:{line_no}")
                    if cls != int(cls) or not 0 <= int(cls) < len(CLASSES):
                        raise ValueError(f"Classe inválida: {label}:{line_no}")
                    if not (
                        0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1
                    ):
                        raise ValueError(f"Coordenadas inválidas: {label}:{line_no}")
                    raw = (
                        math.floor((cx - bw / 2) * w),
                        math.floor((cy - bh / 2) * h),
                        math.ceil((cx + bw / 2) * w),
                        math.ceil((cy + bh / 2) * h),
                    )
                    box = (
                        max(0, raw[0]),
                        max(0, raw[1]),
                        min(w, raw[2]),
                        min(h, raw[3]),
                    )
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError(f"Caixa vazia: {label}:{line_no}")
                    crop = im.crop(box)
                    # Limiar geométrico definido antes de medir resultados, igual nos splits.
                    if min(crop.size) < 12:
                        rejected.append(
                            {
                                "source": str(path.relative_to(args.source)),
                                "line": line_no,
                                "reason": "menor dimensão < 12 pixels",
                                "size": crop.size,
                            }
                        )
                        continue
                    pixel_hash = hashlib.sha256(
                        str(crop.size).encode() + crop.tobytes()
                    ).hexdigest()
                    if pixel_hash in seen:
                        prior = seen[pixel_hash]
                        if prior[0] != split or prior[1] != int(cls):
                            raise ValueError(
                                f"Recorte duplicado entre splits/classes: {path}:{line_no} e {prior}"
                            )
                        rejected.append(
                            {
                                "source": str(path.relative_to(args.source)),
                                "line": line_no,
                                "reason": "recorte idêntico no mesmo split/classe",
                                "size": crop.size,
                            }
                        )
                        continue
                    seen[pixel_hash] = (split, int(cls), path.name, line_no)
                    out = (
                        root
                        / split
                        / CLASSES[int(cls)]
                        / f"{path.stem}__obj{line_no:03d}.png"
                    )
                    crop.save(out)
                    rows.append(
                        {
                            "file": str(out.relative_to(root)),
                            "classe": CLASSES[int(cls)],
                            "split": split,
                            "source": str(path.relative_to(args.source)),
                            "source_sha256": sha(path),
                            "label_sha256": sha(label),
                            "line": line_no,
                            "x1": box[0],
                            "y1": box[1],
                            "x2": box[2],
                            "y2": box[3],
                            "clipped": raw != box,
                            "width": crop.width,
                            "height": crop.height,
                            "sha256": sha(out),
                            "pixel_sha256": pixel_hash,
                        }
                    )
    with (ev / "manifesto_dataset.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (ev / "recortes_descartados.json").write_text(
        json.dumps(rejected, indent=2, ensure_ascii=False)
    )
    (ev / "hashes_origem.json").write_text(json.dumps(source_hashes, indent=2))
    counts = Counter((r["split"], r["classe"]) for r in rows)
    tree = ["aula6_classificacao/"]
    for split in ["train", "validation"]:
        tree.append(f"  {split}/")
        for name in CLASSES:
            assert counts[split, name] > 0
            tree.append(f"    {name}/ — {counts[split, name]} imagens PNG")
    tree_text = "\n".join(tree) + "\n"
    (ev / "02_estrutura_dataset.txt").write_text(tree_text)
    attribution = """# Dataset de classificação derivado do epi-v1

Origem: https://universe.roboflow.com/pedro-yan-alcantara-palacio/epi-detection-rpi5-6q1f3-qarj6/dataset/1
Licença declarada na exportação original: CC BY 4.0.
Atribuição: dataset epi-detection-rpi5, fornecido pelo professor via Roboflow; fork/exportação do workspace pedro-yan-alcantara-palacio, versão 1.
Transformação realizada: recortes RGB pelas caixas YOLO, salvos como PNG sem redimensionamento; coordenadas arredondadas para fora e limitadas à imagem; menor dimensão aceita: 12 pixels.
train original → train; valid original → validation. O test original não participa do treinamento nem desta validação.
Classes descrevem o alvo da caixa: uma Pessoa pode vestir Colete e Capacete. O classificador aprende o enquadramento do recorte, não ausência/presença exclusiva de EPI na cena.
Não houve revisão semântica individual de todas as anotações; rótulos foram herdados da fonte. Avaliação por recorte não equivale a teste em pessoas/cenas independentes.
Confira o manifesto, descartes e relatório de auditoria antes de interpretar as métricas.
"""
    (root / "README.md").write_text(attribution)
    (root / "manifesto_dataset.csv").write_bytes(
        (ev / "manifesto_dataset.csv").read_bytes()
    )
    for name in CLASSES:
        selected = []
        for split in ["train", "validation"]:
            group = [r for r in rows if r["classe"] == name and r["split"] == split]
            selected += [group[round(i * (len(group) - 1) / 11)] for i in range(12)]
        canvas = Image.new("RGB", (960, 4 * 160), "white")
        draw = ImageDraw.Draw(canvas)
        for i, r in enumerate(selected):
            with Image.open(root / r["file"]) as im:
                tile = ImageOps.contain(im, (155, 132))
                x = i % 6 * 160
                y = i // 6 * 160
                canvas.paste(tile, (x, y))
                draw.text(
                    (x, y + 134),
                    f"{r['split']} {r['width']}x{r['height']}",
                    fill="black",
                )
        canvas.save(ev / f"amostras_{name}.jpg")
    with ZipFile(args.output / "dataset_aula6_colab.zip", "w", ZIP_DEFLATED) as z:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root.parent))
    assert all(sha(args.source / p) == digest for p, digest in source_hashes.items())
    print(tree_text)
    print(
        "Recortes:",
        len(rows),
        "Descartados:",
        len(rejected),
        "Caixas limitadas:",
        sum(r["clipped"] for r in rows),
    )
    print("ZIP SHA256:", sha(args.output / "dataset_aula6_colab.zip"))
    print("Arquivos de origem conferidos e preservados:", len(source_hashes))


if __name__ == "__main__":
    main()
