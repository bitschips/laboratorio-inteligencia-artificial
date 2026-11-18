# Experimento 001 — Una neurona aprende la puerta lógica AND

**Laboratorio personal de inteligencia artificial**  
**Tecnologías:** Python, NumPy, TensorFlow/Keras y Matplotlib  
**Entorno recomendado:** Google Colab con CPU  
**Cuaderno asociado:** [`001_perceptron_and.ipynb`](01_fundamentos/001_perceptron_and.ipynb)  
**Nivel:** iniciación, con explicación matemática paso a paso

---

## 1. ¿Qué queremos aprender?

El objetivo no consiste en programar directamente una puerta lógica AND, sino en **crear una neurona artificial que descubra, a partir de ejemplos, cómo debe comportarse**.

La puerta AND recibe dos valores (`0` o `1`) y devuelve `1` **solo cuando las dos entradas valen `1`**:

| Entrada `x₁` | Entrada `x₂` | Salida correcta `y` |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

Un programa convencional podría resolverlo con `if x1 == 1 and x2 == 1`. **Aquí no escribimos esa regla**: la red debe aprender parámetros numéricos que produzcan el mismo resultado.

Al terminar podrás distinguir:

1. **Datos:** ejemplos con entradas y soluciones conocidas.
2. **Modelo:** una fórmula con parámetros ajustables.
3. **Entrenamiento:** proceso que modifica esos parámetros para reducir errores.
4. **Inferencia o predicción:** utilización de los parámetros aprendidos, sin entrenar de nuevo.

> **Pregunta previa:** si la entrada es `[1, 0]`, ¿qué salida debería producir el modelo? Respuesta esperada: `0`.

---

## 2. ¿Por qué empezar con AND?

Es uno de los conjuntos de datos más pequeños posibles: solo tenemos **cuatro combinaciones**. Podemos comprobar todas las respuestas manualmente.

Además, AND se puede aprender mediante **una neurona de salida con activación sigmoide**, porque los ejemplos de salida `0` pueden separarse del único caso de salida `1` mediante una frontera lineal.

No estamos creando todavía un sistema de visión artificial. Más adelante, **cada píxel de una imagen podrá convertirse en una entrada**, pero primero conviene entender qué hace una sola neurona.

### Un esquema del experimento

```text
       Entrada x₁ ──(peso w₁)──┐
                              │
                              ▼
                          [ SUMA + b ] ──> [ SIGMOIDE ] ──> p
                              ▲                              │
       Entrada x₂ ──(peso w₂)──┘                              ▼
                                                ¿p >= 0,5?  1 : 0
```

- `w₁`, `w₂`: **pesos** aprendidos.
- `b`: **sesgo** aprendido.
- `p`: resultado de la neurona, entre `0` y `1`.
- `0,5`: umbral que elegimos para transformar el resultado numérico en una clase.

**Precisión terminológica:** el ejemplo usa una *neurona sigmoide entrenada con Keras*, no exactamente el perceptrón clásico de Rosenblatt, que utiliza una función escalón y otra regla de aprendizaje. La idea de combinar entradas, pesos y sesgo es común a ambos.

---

## 3. Preparar Google Colab e importar bibliotecas

Entra en [Google Colab](https://colab.research.google.com/), abre el cuaderno `001_perceptron_and.ipynb` y elige un entorno con **CPU**.

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Ayuda a repetir el experimento con resultados parecidos.
tf.keras.utils.set_random_seed(7)

print("TensorFlow:", tf.__version__)
print("Dispositivos CPU:", tf.config.list_physical_devices("CPU"))
```

### Explicación línea por línea

**`import numpy as np`** importa NumPy, una biblioteca para manejar datos numéricos como matrices. La abreviatura `np` es una convención de Python.

**`import tensorflow as tf`** carga TensorFlow. En este experimento utilizamos su API de alto nivel, **Keras**, para construir y entrenar la neurona.

**`import matplotlib.pyplot as plt`** permite dibujar gráficas: veremos si el error disminuye a medida que aprende.

**`tf.keras.utils.set_random_seed(7)`** establece una semilla para los elementos aleatorios del experimento, como la inicialización de los pesos. Con el mismo entorno ayuda a repetir resultados, aunque **no garantiza reproducibilidad idéntica** en todas las plataformas y configuraciones.

**`tf.__version__`** indica la versión instalada. Es útil anotarla porque los resultados y comportamientos pueden variar con las versiones.

**`tf.config.list_physical_devices("CPU")`** muestra las CPU que TensorFlow reconoce. No entrena nada: únicamente consulta los dispositivos.

### ¿Por qué no usar una TPU de Google?

En este ejemplo hay dos entradas, una neurona y cuatro muestras. El coste de preparar la TPU sería mucho mayor que el cálculo que necesitamos. **La CPU es la opción adecuada**. Más adelante podremos comparar CPU, GPU y TPU con redes e imágenes mayores.

---

## 4. Los datos de entrenamiento: `X` e `y`

```python
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=np.float32)

y = np.array([[0], [0], [0], [1]], dtype=np.float32)

print("Forma de X:", X.shape)
print("Forma de y:", y.shape)
```

### ¿Qué significa `X`?

`X` es la **matriz de características de entrada** (*features*). Cada fila es un ejemplo y cada columna una variable:

```text
X = [
  [0, 0],  ← ejemplo 1
  [0, 1],  ← ejemplo 2
  [1, 0],  ← ejemplo 3
  [1, 1]   ← ejemplo 4
]
```

Por eso `X.shape` devuelve **`(4, 2)`**: cuatro ejemplos, con dos entradas cada uno.

### ¿Qué significa `y`?

`y` contiene las **etiquetas**, es decir, las respuestas correctas correspondientes a cada fila de `X`:

```text
y = [
  [0],  ← respuesta a [0, 0]
  [0],  ← respuesta a [0, 1]
  [0],  ← respuesta a [1, 0]
  [1]   ← respuesta a [1, 1]
]
```

`y.shape` devuelve **`(4, 1)`**: cuatro ejemplos, cada uno con una respuesta.

La **correspondencia por posición** es fundamental: el primer elemento de `y` responde a la primera fila de `X`, y así sucesivamente. Si desordenaras `y` sin desordenar `X`, enseñarías respuestas incorrectas.

### ¿Por qué `dtype=np.float32`?

Indica que los números se almacenan como flotantes de 32 bits. Aunque aquí sean `0` y `1`, TensorFlow suele realizar los cálculos internos con números decimales. Este tipo ofrece un buen equilibrio entre memoria, rendimiento y precisión.

**Comprueba:** ¿qué ocurriría si eliminas una fila de `X` pero no su etiqueta correspondiente en `y`? El número de muestras dejaría de coincidir y no se podría entrenar correctamente.

---

## 5. Crear el modelo con una sola neurona

```python
modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

modelo.summary()
```

### `tf.keras.Sequential([...])`

Crea un modelo en el que las capas se procesan **en secuencia**: la salida de una etapa pasa a la siguiente. En este caso hay una entrada y una única capa de cálculo.

### `tf.keras.Input(shape=(2,))`

Especifica que **cada ejemplo tiene dos números de entrada**. El `2` se refiere a las características de cada muestra, *no* al número de ejemplos.

- `shape=(2,)`: dos características por muestra.
- El tamaño del lote de ejemplos (*batch*) no se fija en esta declaración.
- Si más adelante utilizas una imagen de `4 × 4` píxeles en escala de grises aplanada, tendrás **16 entradas**, no 2.

### `tf.keras.layers.Dense(1, activation="sigmoid")`

Esta es la instrucción principal:

- **`Dense`** crea una capa totalmente conectada.
- **`1`** significa **una neurona** de salida.
- **`activation="sigmoid"`** aplica una función matemática al resultado de la suma ponderada.

La neurona dispone de **tres parámetros entrenables**:

| Parámetro | Para qué sirve |
|---|---|
| `w₁` | Multiplica la entrada `x₁` |
| `w₂` | Multiplica la entrada `x₂` |
| `b` | Desplaza el resultado, incluso cuando las entradas valen cero |

Por eso `modelo.summary()` debería mostrar **3 parámetros**: dos pesos y un sesgo.

No hay capas ocultas: estamos construyendo el modelo más simple que puede aprender AND.

---

## 6. ¿Qué cálculo realiza realmente la neurona?

En primer lugar suma entradas multiplicadas por pesos, más un sesgo:

\[
z = x_1w_1 + x_2w_2 + b
\]

Después aplica la función **sigmoide**:

\[
p=\sigma(z)=\frac{1}{1+e^{-z}}
\]

El valor `p` se encuentra entre `0` y `1`. Podemos interpretarlo como la probabilidad estimada por el modelo de pertenecer a la clase `1`, aunque una salida de este tipo **no garantiza una probabilidad perfectamente calibrada**.

Para convertirlo en una respuesta binaria:

\[
\hat y=\begin{cases}1,& p\ge 0{,}5\\0,&p<0{,}5\end{cases}
\]

### Ejemplo manual: pesos ficticios

Supongamos que una neurona ya ha aprendido estos valores **ilustrativos**:

```text
w₁ = 10
w₂ = 10
b  = -15
```

Entonces:

| Entradas | `z = 10·x₁ + 10·x₂ − 15` | Sigmoide aproximada | Clase |
|---|---:|---:|:---:|
| `[0,0]` | −15 | 0,0000003 | 0 |
| `[0,1]` | −5 | 0,0067 | 0 |
| `[1,0]` | −5 | 0,0067 | 0 |
| `[1,1]` | 5 | 0,9933 | 1 |

**Los valores anteriores no son resultados medidos del entrenamiento**, sino un ejemplo escogido para entender el cálculo. Tus pesos reales pueden ser distintos.

Observa el papel del **sesgo negativo**: aunque una entrada tenga un peso positivo, solo al activar **ambas** el resultado de la suma supera el umbral y la salida se aproxima a `1`.

### Calcula la sigmoide sin TensorFlow

```python
import numpy as np

x1, x2 = 1, 1
w1, w2, b = 10, 10, -15
z = x1 * w1 + x2 * w2 + b
p = 1 / (1 + np.exp(-z))
print("Suma ponderada:", z)
print("Salida:", p)
```

---

## 7. Preparar el entrenamiento: `compile()`

```python
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

`compile()` **configura** cómo se entrenará el modelo. Todavía **no lo entrena**.

### 7.1. `optimizer=Adam(...)`: ¿cómo corregir los pesos?

Un **optimizador** modifica los pesos y el sesgo a partir del error calculado durante el entrenamiento. La idea general es mover esos parámetros en direcciones que reduzcan la pérdida.

Para comprender el principio, piensa en una versión muy sencilla de descenso de gradiente:

\[
w_{nuevo}=w_{actual}-\eta\,\frac{\partial L}{\partial w}
\]

- `L` es la pérdida que queremos reducir.
- La **derivada** indica cómo cambia la pérdida cuando cambia ese peso.
- `η` (*eta*) es la **tasa de aprendizaje** (*learning rate*).

**Adam** utiliza estimaciones adaptativas de los gradientes, por lo que su actualización real tiene más pasos que esta fórmula simplificada. Aun así, el principio es el mismo: **ajustar parámetros usando el error**.

En este cuaderno ponemos `learning_rate=0.1`, una tasa deliberadamente elevada para un problema minúsculo. **No es la tasa recomendada universalmente**. Para otros problemas podría ser inestable y convendría probar valores como `0.001`.

### 7.2. `loss="binary_crossentropy"`: ¿cuánto nos equivocamos?

La **función de pérdida** mide si la probabilidad que predice el modelo encaja con la respuesta correcta.

Para una muestra:

\[
L=-\big[y\ln(p)+(1-y)\ln(1-p)\big]
\]

Ejemplos:

| Respuesta correcta | Predicción | Pérdida aproximada | Interpretación |
|---|---:|---:|---|
| `1` | `0.9` | `0.105` | Buena predicción |
| `1` | `0.2` | `1.609` | Predicción mala |
| `0` | `0.1` | `0.105` | Buena predicción |
| `0` | `0.8` | `1.609` | Predicción mala |

Cuanto más se acerca la probabilidad a la respuesta correcta, **menor es la pérdida**.

En `fit()`, Keras obtiene un valor de pérdida agregado de las muestras del lote para que el optimizador pueda actuar sobre los parámetros.

### 7.3. `metrics=["accuracy"]`: ¿cuántos aciertos hay?

`accuracy` expresa la proporción de clasificaciones correctas. En nuestro modelo binario, las probabilidades se comparan con el umbral `0.5`.

Si acierta **3 de las 4 filas**, la precisión es:

\[
\text{accuracy}=\frac{3}{4}=0{,}75=75\,\%
\]

**La pérdida y la precisión no son lo mismo.** Un modelo puede seguir reduciendo la pérdida aunque ya clasifique correctamente los cuatro casos.

---

## 8. Entrenar de verdad: `fit()`

```python
history = modelo.fit(
    X,
    y,
    epochs=300,
    verbose=0
)

print("Pérdida final:", round(history.history["loss"][-1], 5))
print("Precisión final:", round(100 * history.history["accuracy"][-1], 1), "%")
```

Ahora sí ocurre el **aprendizaje**. TensorFlow ajusta automáticamente los pesos y el sesgo.

### ¿Qué pasa dentro de una época?

Simplificando, TensorFlow realiza estos pasos:

1. Toma las entradas `X` y produce predicciones con los pesos actuales (**pasada hacia delante**).
2. Compara las predicciones con `y` mediante la función de pérdida.
3. Calcula cómo influye cada parámetro en el error usando **diferenciación automática y retropropagación**.
4. El optimizador **Adam** actualiza los pesos y el sesgo.
5. Registra métricas del entrenamiento.

Una **época** es una pasada completa por los datos de entrenamiento. Como aquí solo existen cuatro muestras y el tamaño de lote predeterminado de Keras es mayor que cuatro, normalmente se procesan juntas en **un lote por época**. Con `epochs=300` hacemos 300 pasadas por los cuatro casos.

### ¿Qué significan los argumentos?

- `X`: entradas.
- `y`: salidas correctas.
- `epochs=300`: número de épocas.
- `verbose=0`: evita imprimir el progreso época por época; **no desactiva el entrenamiento**.

### ¿Qué es `history`?

El objeto devuelto por `fit()` conserva la evolución de las métricas:

```python
history.history["loss"]       # Pérdida de cada época.
history.history["accuracy"]   # Precisión de cada época.
```

Los índices de Python empiezan en cero; `[-1]` selecciona el **último elemento**, es decir, el valor registrado al finalizar el entrenamiento.

**Atención:** ejecutar otra vez `modelo.fit(...)` sin reconstruir el modelo **continúa entrenando desde los pesos actuales**, no desde cero. Para comparar experimentos de forma justa, crea de nuevo el modelo y configura de nuevo `compile()` antes de cada entrenamiento.

---

## 9. Representar el proceso con gráficas

```python
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="Error (loss)")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Evolución del error")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="Precisión (accuracy)")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.ylim(0, 1.05)
plt.title("Aciertos durante el entrenamiento")
plt.grid(True)
plt.legend()
plt.show()
```

### Gráfica de `loss`

- **Eje horizontal:** época del entrenamiento.
- **Eje vertical:** pérdida obtenida.
- Lo esperable es que **la curva descienda** conforme el modelo encuentra parámetros mejores.
- No tiene por qué llegar exactamente a cero.

### Gráfica de `accuracy`

- **Eje horizontal:** época.
- **Eje vertical:** proporción de aciertos entre `0` y `1`.
- En solo cuatro muestras, cada error representa un **25 %** del conjunto.
- Si llega a `1.0`, el modelo clasifica bien **los cuatro ejemplos de entrenamiento**.

> **Limitación importante:** en este experimento entrenamos y comprobamos los mismos cuatro ejemplos. Una precisión del 100 % **no demuestra capacidad de generalización** a datos nuevos. En AND binario hemos incluido todos los casos posibles; en problemas reales hay muchísimos más ejemplos y conviene reservar datos separados para validar y evaluar.

---

## 10. Utilizar lo aprendido: `predict()`

```python
probabilidades = modelo.predict(X, verbose=0)

print("Entradas | Probabilidad | Clase | Esperado")
for entradas, prob, esperado in zip(X, probabilidades, y):
    clase = 1 if prob[0] >= 0.5 else 0
    print(entradas.astype(int), float(prob[0]), clase, int(esperado[0]))
```

### ¿Qué hace `predict()`?

Utiliza los **pesos y el sesgo ya entrenados** para calcular nuevas salidas: este proceso se denomina **inferencia**.

A diferencia de `fit()`, **`predict()` no actualiza los parámetros**.

La variable `probabilidades` tiene la forma `(4, 1)`, porque hemos pasado cuatro ejemplos y el modelo entrega una salida para cada uno.

### ¿Qué significa `prob[0] >= 0.5`?

Por ejemplo, si la neurona produce `0.91`:

```python
0.91 >= 0.5  # True → clase 1
```

Y si produce `0.04`:

```python
0.04 >= 0.5  # False → clase 0
```

La expresión `1 if ... else 0` aplica esa decisión. **La clasificación por umbral es una elección nuestra**: la neurona entrega un valor continuo entre 0 y 1.

### ¿Qué hace el bucle `for`?

- `zip(X, probabilidades, y)` recorre juntas las entradas, sus probabilidades y las respuestas esperadas.
- `entradas.astype(int)` muestra los valores como enteros (`0` y `1`) para que sean más legibles.
- `float(prob[0])` transforma un valor numérico de NumPy en un número Python fácil de imprimir.
- `int(esperado[0])` muestra la clase correcta como `0` o `1`.

### Una predicción individual

```python
nueva_entrada = np.array([[1, 0]], dtype=np.float32)
p = float(modelo.predict(nueva_entrada, verbose=0)[0, 0])
print("Probabilidad:", p)
print("Clase predicha:", int(p >= 0.5))
```

¿Por qué `[[1, 0]]` tiene **dos corchetes**? Porque Keras espera una **matriz de muestras**. Aunque solo haya una, su forma debe ser `(1, 2)`: una fila, dos características.

`[0, 0]` al final de `predict(...)` selecciona la **primera fila y la primera salida** de la matriz resultante.

---

## 11. Inspeccionar los pesos que aprendió

```python
pesos, sesgo = modelo.layers[0].get_weights()

print(f"Peso 1: {pesos[0, 0]:.4f}")
print(f"Peso 2: {pesos[1, 0]:.4f}")
print(f"Sesgo:  {sesgo[0]:.4f}")
```

### ¿Qué obtenemos?

- `modelo.layers[0]` accede a la capa `Dense` del modelo.
- `get_weights()` devuelve los parámetros guardados en esa capa.
- `pesos` es una matriz de forma `(2, 1)`.
- `sesgo` es un vector de forma `(1,)`.

Lo que el modelo ha aprendido es esencialmente **una fórmula con tres números**. No se ha creado un `if` interno que contenga literalmente la tabla de verdad AND.

### Comprobar un resultado a mano

```python
z = pesos[0, 0] * 1 + pesos[1, 0] * 1 + sesgo[0]
prob_manual = 1 / (1 + np.exp(-z))
print("Probabilidad calculada manualmente:", prob_manual)

prob_tensorflow = modelo.predict(
    np.array([[1, 1]], dtype=np.float32), verbose=0
)[0, 0]
print("Probabilidad según TensorFlow:", prob_tensorflow)
```

Los dos resultados deberían ser **muy próximos**, salvo pequeñas diferencias por precisión numérica.

Esta es una de las comprobaciones más útiles del experimento: demuestra qué operaciones hay detrás de `Dense(1, activation="sigmoid")`.

---

## 12. Entrenamiento frente a predicción: tiempo y recursos

| Etapa | Método | ¿Cambia pesos? | Trabajo principal |
|---|---|:---:|---|
| Crear el modelo | `Sequential()` | Inicializa parámetros | Construir la arquitectura |
| Configurar | `compile()` | No los aprende | Elegir optimizador, pérdida y métricas |
| Entrenar | `fit()` | **Sí** | Predecir, calcular error, gradientes y actualizar pesos |
| Utilizar | `predict()` | **No** | Calcular salidas con los parámetros ya aprendidos |
| Inspeccionar | `get_weights()` | No | Leer pesos y sesgo |

En un ejemplo tan pequeño, las mediciones de tiempo pueden estar dominadas por el inicio de TensorFlow y otros costes del entorno. **No es una prueba representativa del rendimiento de una TPU**.

---

## 13. Experimentos de ampliación

### Prueba A. ¿Qué ocurre si entrenas menos?

1. Cambia `epochs=300` por `epochs=10`.
2. **Vuelve a ejecutar desde la celda donde se crea `modelo`**.
3. Compara pérdida, precisión y predicciones.
4. Repite con 30, 100 y 300 épocas.

**Pregunta:** ¿a partir de cuántas épocas clasifica bien las cuatro combinaciones? El punto exacto puede variar.

### Prueba B. Cambiar AND por OR

La puerta OR devuelve `1` cuando **al menos una entrada** vale `1`.

```python
y = np.array([[0], [1], [1], [1]], dtype=np.float32)
```

Recrea el modelo y vuelve a entrenar. Una sola neurona sigmoide también puede aprender OR.

### Prueba C. Intentar aprender XOR

La puerta XOR devuelve `1` cuando las entradas son **diferentes**:

| `x₁` | `x₂` | XOR |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```python
y = np.array([[0], [1], [1], [0]], dtype=np.float32)
```

**¿Por qué resulta problemático?** Las cuatro combinaciones XOR **no son linealmente separables**. Una única neurona de salida sigmoide, seguida de un umbral, solo construye una frontera de decisión lineal; por eso **no puede clasificar correctamente las cuatro combinaciones XOR a la vez**.

Este será un buen siguiente experimento para introducir **capas ocultas** y comprobar cómo permiten modelar fronteras no lineales.

### Prueba D. Modificar la tasa de aprendizaje

Compara `learning_rate=0.1`, `0.01` y `0.001`. Reconstruye el modelo en cada prueba y mantén las demás condiciones lo más iguales posible.

**Pregunta:** ¿qué cambia cuando la tasa de aprendizaje es mucho menor? Observa las curvas de pérdida después del mismo número de épocas.

---

## 14. Errores habituales y cómo resolverlos

| Problema | Causa probable | Solución |
|---|---|---|
| `NameError: modelo is not defined` | No se ejecutó la celda de creación | Ejecuta las celdas en orden |
| `NameError: X is not defined` | Faltan los datos | Ejecuta la celda de `X` e `y` |
| Las predicciones cambian tras otra ejecución | El modelo se ha vuelto a entrenar o se reinicializó | Reconstruye desde cero para comparar |
| Los datos tienen dimensiones incompatibles | Número distinto de muestras o forma errónea | Comprueba `X.shape` e `y.shape` |
| Aparece un número como `0.973` y no `1` | La neurona produce una salida sigmoide | Aplica el umbral `>= 0.5` |
| `accuracy=1.0`, pero `loss` aún es positiva | Acierta las clases, pero las probabilidades no son perfectas | Es normal: las dos métricas miden cosas distintas |
| Un experimento posterior conserva pesos de uno anterior | Reutilizaste el mismo objeto `modelo` | Crea un nuevo modelo y compílalo antes de entrenar |
| TensorFlow no está disponible en Colab | Entorno o instalación incompleta | Prueba `%pip install tensorflow` y reinicia el entorno si procede |

---

## 15. Cuestionario de autoevaluación

1. ¿Qué diferencia hay entre la matriz `X` y el vector o matriz `y`?
2. ¿Por qué la forma de `X` es `(4, 2)` y la de `y` es `(4, 1)`?
3. ¿Cuántos parámetros entrenables tiene nuestra neurona? ¿Cuáles son?
4. ¿Qué representa `b` en `x₁w₁ + x₂w₂ + b`?
5. ¿Qué devuelve la función sigmoide y por qué hace falta un umbral?
6. ¿En qué se diferencian `compile()`, `fit()` y `predict()`?
7. ¿Qué significa `epochs=300`?
8. ¿Por qué es posible que la precisión llegue al 100 % mientras la pérdida continúa disminuyendo?
9. ¿Qué ocurre si vuelves a ejecutar `fit()` sin reconstruir el modelo?
10. ¿Por qué una única neurona sigmoide puede aprender AND, pero no XOR?

### Respuestas orientativas

1. `X` contiene entradas; `y`, las respuestas esperadas.
2. Cuatro muestras con dos valores de entrada y una respuesta por muestra.
3. Tres: dos pesos y un sesgo.
4. Desplaza la suma ponderada y ayuda a ajustar el umbral de decisión.
5. Un valor entre 0 y 1; el umbral lo convierte en clase `0` o `1`.
6. `compile()` prepara, `fit()` entrena y `predict()` usa lo aprendido.
7. Trescientas pasadas completas por el conjunto de entrenamiento.
8. Acertar la clase no implica asignar probabilidades casi exactas.
9. Sigue entrenando con los pesos actuales.
10. AND es linealmente separable; XOR no.

---

## 16. Glosario

| Concepto | Significado |
|---|---|
| **Muestra** | Un caso de entrada usado para entrenar o evaluar |
| **Característica** (*feature*) | Un valor de entrada de cada muestra |
| **Etiqueta** (*label*) | La respuesta correcta asociada a una muestra |
| **Neurona artificial** | Operación de suma ponderada, sesgo y función de activación |
| **Peso** | Número que multiplica una entrada y se aprende durante el entrenamiento |
| **Sesgo** (*bias*) | Parámetro adicional que ajusta la respuesta de la neurona |
| **Activación** | Función aplicada a la suma ponderada de la neurona |
| **Sigmoide** | Función que transforma un número en otro entre 0 y 1 |
| **Parámetro entrenable** | Valor que el optimizador puede modificar |
| **Modelo** | Estructura de cálculo formada por capas y parámetros |
| **Función de pérdida** | Medida numérica de los errores del modelo |
| **Optimizador** | Algoritmo que ajusta parámetros para reducir la pérdida |
| **Gradiente** | Información sobre cómo cambia la pérdida al variar un parámetro |
| **Época** | Una pasada completa por todos los datos de entrenamiento |
| **Accuracy** | Proporción de clasificaciones correctas |
| **Inferencia** | Uso del modelo entrenado para obtener resultados |
| **Generalización** | Capacidad de comportarse bien con casos no utilizados para entrenar |
| **TPU** | Procesador especializado en operaciones de aprendizaje automático |

---

## 17. Plantilla para guardar tus conclusiones

Completa esta ficha después de ejecutar el cuaderno:

```markdown
### Resultados del EXP-001

- **Fecha de ejecución:**
- **Versión de TensorFlow:**
- **Hardware utilizado:** CPU
- **Número de épocas:** 300
- **Tasa de aprendizaje:** 0.1
- **Pérdida final:**
- **Precisión final (entrenamiento):**
- **Pesos aprendidos:**
- **Sesgo aprendido:**
- **Resultado para [1, 0]:**
- **¿Ha aprendido la función AND?:**
- **Qué he entendido mejor:**
- **Qué dudas me quedan:**
- **Próximo experimento:**
```

**Conclusión central:** TensorFlow no recibe la regla AND escrita como una condición. Recibe datos de entrada, las respuestas correctas y una arquitectura matemática; mediante el entrenamiento **ajusta dos pesos y un sesgo**. Una vez ajustados, `predict()` puede utilizar esos parámetros sin volver a entrenar.

---

**Para guardar este material en GitHub:** coloca este archivo junto al cuaderno, por ejemplo en `01_fundamentos/001_perceptron_and_explicacion_detallada.md`, y añade desde el `README.md` del repositorio un enlace a esta explicación.