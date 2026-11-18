# laboratorio-inteligencia-artificial
Laboratory for learning AI

**Cuadernos de Google Colab · TensorFlow · Visión artificial · CPU, GPU y TPU**

Este repositorio es mi **índice personal de experimentos con inteligencia artificial**. La idea es guardar pruebas pequeñas, entender qué hace cada línea de código y poder retomarlas más adelante sin tener que buscarlas entre conversaciones y cuadernos sueltos.

> **Cómo usarlo:** desde este índice localizo un experimento, abro su cuaderno en Google Colab, ejecuto el código y anoto lo que he aprendido. Los enlaces a los cuadernos se añadirán cuando estén subidos al repositorio.

## Índice de experimentos

| ID | Área | Experimento | Cuaderno | Estado |
|:--:|---|---|---|---|
| 001 | Fundamentos | Perceptrón: puerta lógica AND | `01_fundamentos/001_perceptron_and.ipynb` | Por enlazar |
| 002 | TensorFlow | `compile()`, `fit()` y `predict()` paso a paso | `02_tensorflow/002_entrenar_y_predecir.ipynb` | Por enlazar |
| 003 | Visión artificial | Reconocer patrones en imágenes de 4 × 4 píxeles | `03_vision_artificial/003_imagenes_4x4.ipynb` | Por enlazar |
| 004 | Redes neuronales | Número de neuronas y capas ocultas | `02_tensorflow/004_capas_y_neuronas.ipynb` | Por enlazar |
| 005 | Aprendizaje automático | Regresión y representación de resultados | `01_fundamentos/005_regresion.ipynb` | Por enlazar |
| 006 | Aprendizaje por refuerzo | Laberinto 5 × 5 y Q-learning | `05_proyectos/006_laberinto_q_learning.ipynb` | Por enlazar |
| 007 | Aceleración | Comparar tiempos de CPU, GPU y TPU | `04_gpu_tpu/007_cpu_gpu_tpu.ipynb` | Pendiente |
| 008 | Visión artificial | Clasificación de imágenes de habitaciones | `03_vision_artificial/008_habitaciones.ipynb` | Idea |

**Nota:** los nombres de cuadernos son una propuesta de organización. Hasta que se añadan los archivos `.ipynb`, no hay enlaces funcionales a ellos.

## Carpetas del laboratorio

```text
laboratorio-inteligencia-artificial/
├── README.md
├── 01_fundamentos/
│   ├── 001_perceptron_and.ipynb
│   └── 005_regresion.ipynb
├── 02_tensorflow/
│   ├── 002_entrenar_y_predecir.ipynb
│   └── 004_capas_y_neuronas.ipynb
├── 03_vision_artificial/
│   ├── 003_imagenes_4x4.ipynb
│   └── 008_habitaciones.ipynb
├── 04_gpu_tpu/
│   └── 007_cpu_gpu_tpu.ipynb
└── 05_proyectos/
    └── 006_laberinto_q_learning.ipynb
```

*El árbol muestra la estructura prevista; no indica que los cuadernos ya estén publicados.*

## Ficha recomendada para cada experimento

Al comienzo de cada cuaderno, añade una celda **Markdown** con esta información:

```markdown
# EXP-XXX — Título del experimento

- **Fecha:** AAAA-MM-DD
- **Objetivo:** ¿Qué quiero comprender o comprobar?
- **Herramientas:** Python, TensorFlow, NumPy, etc.
- **Hardware:** CPU / GPU / TPU
- **Datos de entrada:** ¿Qué datos utilizo?
- **Modelo:** Capas, neuronas y función de activación.
- **Entrenamiento:** Optimizador, función de pérdida y épocas.
- **Resultado:** ¿Qué ocurrió? ¿Qué precisión y tiempo obtuve?
- **Conclusiones:** ¿Qué he aprendido?
- **Siguiente prueba:** ¿Qué cambiaría?
```

## Cómo abrir cuadernos de GitHub en Colab

Cuando un cuaderno esté publicado en el repositorio, su enlace directo tendrá esta estructura:

```text
https://colab.research.google.com/github/bitschips/laboratorio-inteligencia-artificial/blob/main/CARPETA/ARCHIVO.ipynb
```

Puedes sustituir la columna **Cuaderno** de la tabla por un enlace de Markdown:

```markdown
[Abrir en Colab](https://colab.research.google.com/github/bitschips/laboratorio-inteligencia-artificial/blob/main/03_vision_artificial/003_imagenes_4x4.ipynb)
```

> Ese enlace es **un ejemplo de formato**: solo funcionará cuando exista el repositorio y se haya subido el cuaderno correspondiente. Si utilizas otro nombre de repositorio, cambia la URL.

### Flujo de trabajo

1. Crear o abrir el experimento en [Google Colab](https://colab.research.google.com/).
2. Escribir una breve ficha al principio del cuaderno.
3. Ejecutar las pruebas, anotar resultados y conclusiones.
4. Guardar el cuaderno en GitHub desde Colab (**Archivo → Guardar una copia en GitHub**), cuando esté listo.
5. Actualizar su fila en este índice con el enlace real.

## Recordatorios

- Para pruebas diminutas (como imágenes de 4 × 4), **la CPU suele ser suficiente**; GPU y TPU pueden tener costes de inicialización superiores al tiempo de cálculo.
- Un entrenamiento reproducible debería anotar versiones de bibliotecas, datos y semillas aleatorias cuando sea necesario.
- No subir a un repositorio público contraseñas, claves API, tokens ni imágenes privadas.

---

**Estado del laboratorio:** índice inicial · **Última revisión:** 18/11/2026
