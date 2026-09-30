#!/usr/bin/env python3
# Copyright 2026 Alex Díaz
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODELOS_DIR = Path(__file__).resolve().parent / "modelos"
MODELOS = {
    "yunet": (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    ),
    "sface": (
        "face_recognition_sface_2021dec.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
    ),
}
# Umbral recomendado por los autores de SFace para similitud coseno
UMBRAL_POR_DEFECTO = 0.363
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------
def obtener_modelo(clave):
    nombre, url = MODELOS[clave]
    ruta = MODELOS_DIR / nombre
    if not ruta.exists():
        MODELOS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {nombre} ...")
        # Se descarga a un archivo temporal y se renombra al terminar, para que
        # una descarga interrumpida no deje un modelo incompleto.
        temporal = ruta.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, temporal)
            temporal.replace(ruta)
        except BaseException:
            temporal.unlink(missing_ok=True)
            raise
    return str(ruta)


class ReconocedorFacial:
    def __init__(self, umbral_deteccion=0.8):
        self.detector = cv2.FaceDetectorYN.create(
            obtener_modelo("yunet"), "", (320, 320), umbral_deteccion, 0.3, 5000
        )
        self.reconocedor = cv2.FaceRecognizerSF.create(obtener_modelo("sface"), "")

    def detectar(self, img):
        """Devuelve la cara más grande detectada (array de 15 valores) o None.

        YuNet no detecta caras muy grandes (selfies de cerca), así que si no
        encuentra ninguna reintenta con la imagen reducida a 640 y 320 px.
        """
        h, w = img.shape[:2]
        for lado in (None, 640, 320):
            escala = 1.0 if lado is None else lado / max(h, w)
            if escala >= 1 and lado is not None:
                continue
            reducida = img if escala == 1 else cv2.resize(img, (int(w * escala), int(h * escala)))
            self.detector.setInputSize((reducida.shape[1], reducida.shape[0]))
            _, caras = self.detector.detect(reducida)
            if caras is not None and len(caras) > 0:
                cara = max(caras, key=lambda c: c[2] * c[3]).copy()
                cara[:14] /= escala  # coordenadas de vuelta a la imagen original
                return cara
        return None

    def embedding(self, img, cara):
        alineada = self.reconocedor.alignCrop(img, cara)
        return self.reconocedor.feature(alineada).flatten()

    def similitud(self, e1, e2):
        return float(
            self.reconocedor.match(
                e1.reshape(1, -1), e2.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE
            )
        )


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def leer_imagen(ruta, lado_max=1280):
    # imdecode admite rutas con tildes/ñ (cv2.imread falla con ellas en Windows)
    datos = np.fromfile(str(ruta), dtype=np.uint8)
    img = cv2.imdecode(datos, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    escala = lado_max / max(h, w)
    if escala < 1:
        img = cv2.resize(img, (int(w * escala), int(h * escala)))
    return img


def listar_imagenes(carpeta):
    carpeta = Path(carpeta)
    if not carpeta.exists():
        return []
    return sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTENSIONES)


def anotar(img, cara, texto, aceptado):
    color = (0, 200, 0) if aceptado else (0, 0, 230)
    out = img.copy()
    if cara is not None:
        x, y, w, h = cara[:4].astype(int)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        for i in range(5):
            cv2.circle(out, (int(cara[4 + 2 * i]), int(cara[5 + 2 * i])), 2, (255, 255, 0), -1)
    escala = max(0.5, out.shape[1] / 900)
    cv2.rectangle(out, (0, 0), (out.shape[1], int(40 * escala)), (0, 0, 0), -1)
    cv2.putText(out, texto, (8, int(28 * escala)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * escala, color, max(1, int(2 * escala)))
    return out


def crear_plantilla(rf, carpeta_registro):
    embs = []
    for ruta in listar_imagenes(carpeta_registro):
        img = leer_imagen(ruta)
        cara = rf.detectar(img) if img is not None else None
        if cara is None:
            print(f"  [aviso] sin cara en imagen de registro: {ruta.name}")
            continue
        embs.append(rf.embedding(img, cara))
        print(f"  registrada: {ruta.name}")
    if not embs:
        sys.exit("ERROR: ninguna imagen válida en la carpeta de registro.")
    return embs


def puntuar(rf, plantilla, img):
    """Devuelve (cara, similitud máxima frente a la plantilla) o (None, None)."""
    cara = rf.detectar(img)
    if cara is None:
        return None, None
    emb = rf.embedding(img, cara)
    return cara, max(rf.similitud(emb, p) for p in plantilla)


# --------------------------------------------------------------------------
# Métricas
# --------------------------------------------------------------------------
def tasas(genuinas, impostoras, umbral):
    """FAR: impostores aceptados. FRR: genuinos rechazados (sin cara = rechazo)."""
    g = np.array([s if s is not None else -1.0 for s in genuinas])
    i = np.array([s if s is not None else -1.0 for s in impostoras])
    frr = float(np.mean(g < umbral)) if len(g) else float("nan")
    far = float(np.mean(i >= umbral)) if len(i) else float("nan")
    return far, frr


def curva(genuinas, impostoras):
    umbrales = np.linspace(-0.2, 1.0, 601)
    fars, frrs = zip(*(tasas(genuinas, impostoras, t) for t in umbrales))
    fars, frrs = np.array(fars), np.array(frrs)
    k = int(np.argmin(np.abs(fars - frrs)))
    return umbrales, fars, frrs, float(umbrales[k]), float((fars[k] + frrs[k]) / 2)


def graficas(salida, grupos, umbral):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib no instalado: se omiten las gráficas)")
        return

    genuinas = grupos.get("genuino", [])
    todas_falsas = [s for k, v in grupos.items() if k != "genuino" for s in v]

    # Histograma de puntuaciones
    plt.figure(figsize=(9, 5))
    bins = np.linspace(-0.2, 1.0, 49)
    for nombre, v in grupos.items():
        v = [s for s in v if s is not None]
        if v:
            plt.hist(v, bins=bins, alpha=0.55, label=f"{nombre} (n={len(v)})")
    plt.axvline(umbral, color="k", ls="--", label=f"umbral = {umbral:.3f}")
    plt.xlabel("Similitud coseno con la plantilla")
    plt.ylabel("Número de imágenes")
    plt.title("Distribución de puntuaciones")
    plt.legend()
    plt.tight_layout()
    plt.savefig(salida / "histograma.png", dpi=150)
    plt.close()

    if not genuinas or not todas_falsas:
        return
    umbrales, fars, frrs, t_eer, eer = curva(genuinas, todas_falsas)

    # FAR / FRR vs umbral
    plt.figure(figsize=(9, 5))
    plt.plot(umbrales, fars * 100, label="FAR (falsos aceptados)")
    plt.plot(umbrales, frrs * 100, label="FRR (falsos rechazos)")
    plt.axvline(umbral, color="k", ls="--", label=f"umbral usado = {umbral:.3f}")
    plt.scatter([t_eer], [eer * 100], color="r", zorder=5,
                label=f"EER = {eer * 100:.1f}% (umbral {t_eer:.3f})")
    plt.xlabel("Umbral")
    plt.ylabel("%")
    plt.title("FAR y FRR en función del umbral")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida / "far_frr.png", dpi=150)
    plt.close()

    # ROC
    plt.figure(figsize=(6, 6))
    plt.plot(fars * 100, (1 - frrs) * 100)
    plt.plot([0, 100], [0, 100], "k:", alpha=0.4)
    plt.xlabel("FAR (%)")
    plt.ylabel("Tasa de aceptación genuina, 1-FRR (%)")
    plt.title("Curva ROC")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida / "roc.png", dpi=150)
    plt.close()

    # Tasa de aceptación por categoría
    nombres = list(grupos)
    acept = [np.mean([s is not None and s >= umbral for s in grupos[n]]) * 100 for n in nombres]
    plt.figure(figsize=(max(6, 1.3 * len(nombres)), 5))
    colores = ["tab:green" if n == "genuino" else "tab:red" for n in nombres]
    barras = plt.bar(nombres, acept, color=colores)
    plt.bar_label(barras, fmt="%.0f%%")
    plt.ylabel("% de imágenes ACEPTADAS como el usuario")
    plt.title(f"Aceptación por categoría (umbral {umbral:.3f})")
    plt.ylim(0, 110)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(salida / "por_categoria.png", dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------
def cmd_evaluar(args):
    dataset, salida = Path(args.dataset), Path(args.salida)
    (salida / "anotadas").mkdir(parents=True, exist_ok=True)
    rf = ReconocedorFacial()

    print("== Registro del usuario ==")
    plantilla = crear_plantilla(rf, dataset / "registro")

    categorias = {"genuino": dataset / "genuino", "impostores": dataset / "impostores"}
    ataques = dataset / "ataques"
    if ataques.exists():
        for sub in sorted(p for p in ataques.iterdir() if p.is_dir()):
            categorias[f"ataque_{sub.name}"] = sub

    filas, grupos = [], {}
    print("\n== Evaluación ==")
    for cat, carpeta in categorias.items():
        imagenes = listar_imagenes(carpeta)
        if not imagenes:
            continue
        grupos[cat] = []
        for ruta in imagenes:
            img = leer_imagen(ruta)
            if img is None:
                print(f"  [aviso] no se pudo leer {ruta}")
                continue
            t0 = time.perf_counter()
            cara, sim = puntuar(rf, plantilla, img)
            ms = (time.perf_counter() - t0) * 1000
            aceptado = sim is not None and sim >= args.umbral
            grupos[cat].append(sim)

            esperado = "aceptar" if cat == "genuino" else "rechazar"
            correcto = aceptado == (cat == "genuino")
            filas.append({
                "categoria": cat,
                "imagen": ruta.name,
                "cara_detectada": cara is not None,
                "similitud": "" if sim is None else f"{sim:.4f}",
                "decision": "ACEPTADO" if aceptado else "RECHAZADO",
                "esperado": esperado,
                "correcto": correcto,
                "tiempo_ms": f"{ms:.1f}",
            })
            texto = ("SIN CARA" if sim is None else f"{sim:.3f}") + \
                    (" ACEPTADO" if aceptado else " RECHAZADO")
            cv2.imwrite(str(salida / "anotadas" / f"{cat}__{ruta.stem}.jpg"),
                        anotar(img, cara, texto, aceptado))
            print(f"  {cat:<22} {ruta.name:<35} {texto}")

    if not filas:
        sys.exit("ERROR: no hay imágenes para evaluar.")

    with open(salida / "resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    # ---- Métricas ----
    lineas = [f"Umbral de decisión: {args.umbral:.3f} (similitud coseno)",
              f"Imágenes de registro en la plantilla: {len(plantilla)}", ""]
    lineas.append(f"{'Categoría':<24}{'N':>5}{'Sin cara':>10}{'Aceptadas':>11}"
                  f"{'% acept.':>10}{'Media sim.':>12}{'Máx sim.':>10}")
    for cat, v in grupos.items():
        validas = [s for s in v if s is not None]
        acept = sum(s >= args.umbral for s in validas)
        lineas.append(
            f"{cat:<24}{len(v):>5}{len(v) - len(validas):>10}{acept:>11}"
            f"{100 * acept / len(v):>9.1f}%"
            f"{(np.mean(validas) if validas else float('nan')):>12.3f}"
            f"{(max(validas) if validas else float('nan')):>10.3f}")

    genuinas = grupos.get("genuino", [])
    impostoras = grupos.get("impostores", [])
    ataques_todos = [s for k, v in grupos.items() if k.startswith("ataque_") for s in v]
    lineas.append("")
    if genuinas:
        _, frr = tasas(genuinas, [], args.umbral)
        lineas.append(f"FRR  (usuario legítimo rechazado) : {frr * 100:6.2f}%")
    if impostoras:
        far, _ = tasas([], impostoras, args.umbral)
        lineas.append(f"FAR  (impostor aceptado)          : {far * 100:6.2f}%")
    if ataques_todos:
        apcer, _ = tasas([], ataques_todos, args.umbral)
        lineas.append(f"IAPMR (ataques de presentación que "
                      f"engañan al sistema): {apcer * 100:6.2f}%")
    falsas = impostoras + ataques_todos
    if genuinas and falsas:
        _, _, _, t_eer, eer = curva(genuinas, falsas)
        lineas.append(f"EER  (FAR = FRR)                  : {eer * 100:6.2f}% "
                      f"con umbral {t_eer:.3f}")
        total = len(filas)
        aciertos = sum(r["correcto"] for r in filas)
        lineas.append(f"Exactitud global                  : {100 * aciertos / total:6.2f}% "
                      f"({aciertos}/{total})")
    tiempos = [float(r["tiempo_ms"]) for r in filas]
    lineas.append(f"Tiempo medio por imagen           : {np.mean(tiempos):.1f} ms")

    texto = "\n".join(lineas)
    (salida / "metricas.txt").write_text(texto, encoding="utf-8")
    print("\n== Métricas ==\n" + texto)

    graficas(salida, grupos, args.umbral)
    print(f"\nResultados guardados en: {salida.resolve()}")


def abrir_camara(indice):
    cap = cv2.VideoCapture(indice)
    if not cap.isOpened():
        sys.exit(f"ERROR: no se pudo abrir la cámara {indice}.")
    return cap


def cmd_capturar(args):
    """Guarda fotos desde la webcam (útil para ataques: foto en papel, celular, máscara...)."""
    destino = Path(args.destino)
    destino.mkdir(parents=True, exist_ok=True)
    rf = ReconocedorFacial()
    cap = abrir_camara(args.camara)
    n = len(listar_imagenes(destino))
    print("ESPACIO = guardar foto | Q/ESC = salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cara = rf.detectar(frame)
        vista = anotar(frame, cara, f"{destino.name}: {n} fotos  [ESPACIO guardar]", cara is not None)
        cv2.imshow("Capturar", vista)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            n += 1
            ruta = destino / f"{destino.name}_{n:03d}.jpg"
            cv2.imwrite(str(ruta), frame)
            print(f"  guardada {ruta}")
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_camara(args):
    """Verificación en vivo: muestra la similitud con el usuario registrado."""
    rf = ReconocedorFacial()
    plantilla = crear_plantilla(rf, Path(args.dataset) / "registro")
    cap = abrir_camara(args.camara)
    print("Q/ESC = salir")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cara, sim = puntuar(rf, plantilla, frame)
        aceptado = sim is not None and sim >= args.umbral
        texto = "SIN CARA" if sim is None else \
            f"sim={sim:.3f} {'ACEPTADO' if aceptado else 'RECHAZADO'}"
        cv2.imshow("Verificacion facial", anotar(frame, cara, texto, aceptado))
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Evaluación de reconocimiento facial con OpenCV (YuNet + SFace)")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("evaluar", help="Evalúa el sistema con las carpetas del dataset")
    e.add_argument("--dataset", default="dataset")
    e.add_argument("--salida", default="resultados")
    e.add_argument("--umbral", type=float, default=UMBRAL_POR_DEFECTO)
    e.set_defaults(func=cmd_evaluar)

    c = sub.add_parser("capturar", help="Captura fotos desde la webcam a una carpeta")
    c.add_argument("--destino", required=True)
    c.add_argument("--camara", type=int, default=0)
    c.set_defaults(func=cmd_capturar)

    v = sub.add_parser("camara", help="Verificación en vivo con la webcam")
    v.add_argument("--dataset", default="dataset")
    v.add_argument("--camara", type=int, default=0)
    v.add_argument("--umbral", type=float, default=UMBRAL_POR_DEFECTO)
    v.set_defaults(func=cmd_camara)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
