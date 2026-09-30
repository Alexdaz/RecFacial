# Evaluación de reconocimiento facial

Herramienta en Python para medir el rendimiento de un sistema de reconocimiento facial: qué tan bien reconoce al usuario legítimo, si deja pasar a otras personas y si se deja engañar por ataques de presentación (una foto en el celular, una foto impresa o una máscara).

Funciona completamente en local con OpenCV. Las fotos nunca salen de tu computadora.

## Cómo funciona

1. **Detección (YuNet):** localiza la cara y cinco puntos de referencia (ojos, nariz y comisuras de la boca). Si no encuentra ninguna cara, lo vuelve a intentar con la imagen reducida, porque YuNet no detecta caras muy grandes, como las de las selfies tomadas de cerca.
2. **Alineación:** endereza y escala la cara usando esos puntos y la recorta a 112 x 112 px.
3. **Extracción (SFace):** convierte la cara en un vector de 128 números (plantilla biométrica).
4. **Comparación:** calcula la similitud coseno con las fotos de registro. Si es mayor o igual que el umbral (0.363 por defecto), **acepta**; si no, **rechaza**.

**Instrucciones:** [Documentación](/docs/INSTRUCCIONES.md)

## Instalación

Requiere Python 3.9 o superior.

```bash
pip install -r requirements.txt
```

La primera vez que se ejecuta, el programa descarga los modelos (unos 39 MB) en `modelos/`.

## Uso

Organiza tus fotos así:

```
dataset/
├── registro/        3–5 fotos actuales del usuario, de frente y con buena luz
├── genuino/         fotos del usuario en distintas circunstancias
├── impostores/      fotos de otras personas
└── ataques/         una subcarpeta por método
    ├── foto_papel/
    ├── foto_pantalla/
    └── mascara/
```

Formatos admitidos: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, `.tif` y `.tiff`.

```bash
# Evaluar el sistema y generar métricas y gráficas
python recfacial.py evaluar --dataset dataset --salida resultados --umbral 0.4

# Capturar fotos con la webcam (ESPACIO = guardar, Q = salir)
python recfacial.py capturar --destino dataset/ataques/foto_papel

# Verificación en vivo con la webcam
python recfacial.py camara --dataset dataset
```

`--umbral` es opcional en `evaluar` y `camara`, y `--camara N` permite elegir otra webcam.

## Resultados

| Archivo | Contenido |
|---|---|
| `metricas.txt` | FRR, FAR, IAPMR, EER y exactitud |
| `resultados.csv` | Similitud y decisión de cada imagen |
| `histograma.png` | Distribución de puntuaciones por grupo |
| `far_frr.png` | FAR y FRR en función del umbral, con el EER |
| `roc.png` | Curva ROC |
| `por_categoria.png` | Porcentaje de aceptación por categoría |
| `anotadas/` | Cada imagen con el recuadro, los puntos faciales y la puntuación |

**Métricas:**
- **FRR:** porcentaje de veces que se rechaza al usuario legítimo.
- **FAR:** porcentaje de impostores aceptados.
- **IAPMR:** porcentaje de ataques de presentación que engañan al sistema.
- **EER:** punto donde la tasa de falsas aceptaciones y la FRR se igualan.

## Privacidad

Las fotos de otras personas solo deben usarse con su consentimiento. El `.gitignore` excluye `dataset/` y las carpetas de resultados, porque `anotadas/` contiene caras. **No publiques esas carpetas.**

## Licencia

El código de este repositorio se distribuye bajo la [Licencia Apache 2.0](LICENSE). Si redistribuyes este software o un trabajo derivado, debes conservar el archivo [NOTICE](NOTICE) con el crédito al autor.

### Modelos de terceros

Los modelos no se incluyen en el repositorio. Se descargan de [OpenCV Zoo](https://github.com/opencv/opencv_zoo) y se rigen por sus propias licencias:

- **YuNet**: licencia MIT. Wu, W., Peng, H. y Yu, S. (2023). *YuNet: A tiny millisecond-level face detector*. Machine Intelligence Research, 20(5), 656–665. https://doi.org/10.1007/s11633-023-1423-y

- **SFace**: licencia Apache 2.0. Zhong, Y., Deng, W., Hu, J., Zhao, D., Li, X. y Wen, D. (2021). *SFace: Sigmoid-constrained hypersphere loss for robust face recognition*. IEEE Transactions on Image Processing, 30, 2587–2598. https://doi.org/10.1109/TIP.2020.3048632
