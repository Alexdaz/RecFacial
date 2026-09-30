# Actividad 2 (grupal): evaluación de reconocimiento facial

Instrucciones para usar el script de reconocimiento facial, hecho en Python con OpenCV.

---

## 1. Qué hace el programa

El script compara cada foto con las fotos de registro de un usuario y decide si es la misma persona. Lo hace en cuatro pasos:

1. **Detección (YuNet):** encuentra la cara en la imagen y marca 5 puntos: los dos ojos, la nariz y las dos comisuras de la boca.
2. **Alineación:** endereza y escala la cara usando esos 5 puntos y la recorta a 112 x 112 píxeles.
3. **Extracción de características (SFace):** una red neuronal convierte la cara en un vector de 128 números, que es la "plantilla biométrica".
4. **Comparación:** calcula la similitud coseno entre la foto y la plantilla. Si el resultado es **≥ 0.363**, **ACEPTA** si no, **RECHAZA**.

---

## 2. Instalación (una sola vez)

Necesitan Python 3.9 o superior ([python.org](https://www.python.org/downloads/)). En Windows, marquen la opción **"Add Python to PATH"** al instalarlo.

Abran una terminal (en Windows, "Símbolo del sistema" o PowerShell) dentro de la carpeta `reconocimiento_facial` y ejecuten:

**Mac / Linux**
```
pip3 install -r requirements.txt
```

**Windows**
```
pip install -r requirements.txt
```

> Los modelos de OpenCV ya vienen en la carpeta `modelos/`. Si no están, el programa los descarga solo la primera vez.

---

## 3. Preparar las fotos

El equipo hace una prueba, con su cara con un "usuario legítimo". Pongan las fotos en estas carpetas:

```
dataset/
├── registro/        3–5 fotos ACTUALES, de frente y con buena luz
├── genuino/         30 o más fotos suyas en distintas circunstancias
├── impostores/      OTRAS personas con su propia cara  ┐
└── ataques/         OTRAS personas usando su cara      │ 30 o más
    ├── foto_papel/      su foto impresa                   │ en total
    ├── foto_pantalla/   su foto en un celular o tablet    │
    ├── mascara/         una máscara con su cara           │
    └── ...              pueden crear más: maquillaje...   ┘
```

### Las 30 imágenes de suplantación

La actividad pide **al menos 30 imágenes de otros usuarios que intentan falsificar la cara del usuario legítimo, usando diferentes métodos**. El atacante siempre es **otra persona** que intenta hacerse pasar por ustedes, ya sea con su propia cara o con su foto, su pantalla o su máscara. Las 30 se reparten entre `impostores/` y las subcarpetas de `ataques/`. Por ejemplo:

| Carpeta | Ejemplo | Método |
|---|---|---|
| `impostores/` | 12 | Otra persona se presenta con su propia cara |
| `ataques/foto_papel/` | 8 | Otra persona sostiene su foto impresa frente a la cámara |
| `ataques/foto_pantalla/` | 6 | Otra persona muestra su foto o video en un celular |
| `ataques/mascara/` | 4 | Otra persona usa una máscara con su cara |
| **Total** | **30** | 4 métodos |

Aunque en el informe cuenten juntas, **manténganlas en carpetas separadas**. Así el programa calcula por separado el FAR (impostores aceptados) y la IAPMR (ataques que engañan al sistema), y pueden comparar qué método funciona mejor.

### Qué fotos poner en cada carpeta

| Carpeta | Mínimo | Ideas |
|---|---|---|
| `registro/` | 3 | Fotos actuales, de frente, sin lentes de sol y con buena luz. **No las repitan en `genuino/`.** |
| `genuino/` | 30 | Fotos de hace años y actuales, con y sin lentes, barba o maquillaje, con poca luz, a contraluz o con luz lateral, de perfil o inclinadas, con gorra o cubrebocas, selfies y fotos tomadas por otras personas. |
| `impostores/` + `ataques/*/` | 30 entre todas | Ver la tabla anterior. |
| `impostores/` | — | Familiares (sobre todo si se parecen a ustedes), amigos y compañeros de clase, también fotos de famosos que se les parezcan. |
| `ataques/*/` | ~5 por método | Foto en papel, foto o video en pantalla, máscara, foto recortada con los ojos agujereados... Varíen la distancia, el ángulo y la luz en cada método. |

**Consejos:**
- Formatos válidos: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` y `.tif`.
- Pongan nombres de archivo descriptivos para el informe, por ejemplo `2015_lentes.jpg` o `noche_contraluz.jpg`.
- Si en una foto salen varias personas, el programa usa **la cara más grande**. En `genuino/`, recorten las fotos con amigos para que solo salga su cara.

### Capturar los ataques con la webcam

Para los ataques (otra persona sostiene la foto en papel o el celular, o usa la máscara, frente a la cámara) pueden usar el modo captura. Presionen **ESPACIO** para guardar una foto y **Q** para salir:

```
python3 recfacial.py capturar --destino dataset/ataques/foto_papel
```

> [!IMPORTANT]
> El programa evalúa solo **la cara más grande**. Si la cara real de quien sostiene la foto o el celular sale más grande, se evaluaría a esa persona y no el ataque. La foto o el celular debe **taparle la cara** o estar más cerca de la cámara. Antes de presionar ESPACIO, revisen en la vista previa que el recuadro esté sobre la foto.

> En Windows, escriban `python` en lugar de `python3` en todos los comandos.

---

## 4. Ejecutar la evaluación

```
python3 recfacial.py evaluar --dataset dataset
```

Si cada quien guarda los resultados en su propia carpeta, no se mezclan:

```
python3 recfacial.py evaluar --dataset dataset --salida resultados_maria
```

Para probar otro umbral (más alto es más estricto):

```
python3 recfacial.py evaluar --dataset dataset --umbral 0.5
```

### Prueba en vivo con la webcam (opcional, útil para capturas del informe)

```
python3 recfacial.py camara --dataset dataset
```

Muestra en tiempo real la similitud y si los acepta o rechaza. Prueben a poner frente a la cámara la foto impresa, el celular, etc.

---

## 5. Qué genera (carpeta `resultados/`)

| Archivo | Contenido |
|---|---|
| `metricas.txt` | Resumen con todas las métricas (ver abajo) |
| `resultados.csv` | Resultado de cada imagen: similitud, decisión y si acertó. Se abre con Excel. |
| `histograma.png` | Distribución de puntuaciones de genuinos, impostores y ataques |
| `far_frr.png` | FAR y FRR según el umbral, con el punto EER |
| `roc.png` | Curva ROC |
| `por_categoria.png` | Porcentaje de aceptación de cada tipo de prueba |
| `anotadas/` | Cada imagen con el recuadro de la cara, los puntos faciales y la puntuación, para ponerlas como evidencia en el informe |

### Cómo interpretar las métricas

- **FRR (False Rejection Rate):** porcentaje de veces que el sistema **rechaza al usuario legítimo**. Mientras más bajo, mejor.
- **FAR (False Acceptance Rate):** porcentaje de **impostores aceptados**. Mientras más bajo, mejor.
- **IAPMR:** porcentaje de **ataques de presentación** (foto, pantalla, máscara) que **engañan** al sistema.
- **EER (Equal Error Rate):** punto donde FAR y FRR son iguales. Sirve para comparar sistemas: mientras más bajo, mejor.
- **Similitud:** va de -1 a 1. Mientras más cerca de 1, más se parece a la plantilla.

---

## 6. Qué tiene que entregar cada quien al equipo

1. La carpeta `resultados_<nombre>/` completa.
2. Una breve descripción de las fotos usadas: de qué años son, las condiciones de luz y los métodos de ataque.
3. Observaciones: qué fotos genuinas **fallaron** y por qué (luz, edad, ángulo...) y qué ataques **lograron engañar** al sistema.

> [!WARNING] 
> **Privacidad:** las fotos de otras personas (impostores) solo se usan con su permiso. No suban el `dataset/` a sitios públicos.

---

## 7. Puntos que debe cubrir el informe final

Según las pautas de la actividad:

- [ ] Explicar cómo funciona el algoritmo (sección 1 de este documento)
- [ ] Usar 30 o más imágenes del mismo usuario en distintas circunstancias (antiguas y actuales)
- [ ] Usar 30 o más imágenes de otros usuarios que intentan falsificar la cara original (`impostores/` + `ataques/`), con el desglose por método
- [ ] Usar distintos métodos de falsificación (su propia cara, fotos en papel, pantallas, máscaras)
- [ ] Documentar todo y explicar si el sistema resiste bien o mal los ataques y cómo mejorarlo, por ejemplo con detección de vida (parpadeo, movimiento de cabeza), cámaras de profundidad o infrarrojas (como Face ID), análisis de textura de la piel o combinarlo con otro factor (PIN o huella digital)
- [ ] Explicar los factores externos que influyen en el reconocimiento: iluminación, ángulo, distancia, resolución de la cámara, envejecimiento, lentes, barba, maquillaje, cubrebocas, expresión facial
- [ ] Indicar los posibles usos (desbloqueo del celular, pagos, control de acceso, aeropuertos y fronteras, registro de asistencia, banca en línea) y justificar su uso en dispositivos móviles

---

## 8. Problemas frecuentes

| Problema | Solución |
|---|---|
| `ModuleNotFoundError: No module named 'cv2'` | Vuelvan a ejecutar `pip install -r requirements.txt` |
| `python3: command not found` (Windows) | Usen `python` en vez de `python3` |
| "SIN CARA" en muchas fotos | La cara es muy pequeña, está muy de perfil o muy oscura. Si es una prueba genuina, cuenta como rechazo y es un dato válido para el informe. |
| "ninguna imagen válida en la carpeta de registro" | Pongan en `registro/` fotos de frente y bien iluminadas |
| No se abre la webcam | Cerrar cualquier app que utilice la cámara o probar con `--camara 1`. En Mac, den permiso de cámara a la Terminal. |
| No se generan las gráficas | Instalen matplotlib: `pip install matplotlib` |
