# Laboratorio-4

# Análisis de señales elecromiograficas (EMG)

En este laboratorio exploramos señales fisiológicas de EMG utilizando técnicas de adqusicion de datos,filtrado de la señal,aventanamiento y el analisis espectral,teniendo en cuenta las tecnicas vistas en clase.Sabiendo que la EMG es una técnica que mide la actividad eléctrica de los músculos atravez de potenciales de acción.Normalmente se usa para evaluar y ver cómo funcionan los músculos y detectar anomalías en su activación o en la transmisión de señales entre nervios y músculos (Comunicación neuromuscular).[1]

## Requisitos
- **Python 3.9**
- Bibliotecas necesarias:
  - `nidaqmx`
  - `numpy`
  - `matplotlib`

Instalar dependencias:
```bash
pip install nidaqmx numpy matplotlib
```

## Estructura del Código

### 1. Adquisición de datos
1.1 Librerias y definición de parámetros
```python
import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np
```
- nidaqmx: Esta librería permite interactuar con dispositivos de adquisición de datos (DAQ) de National Instruments.[2]
- AcquisitionType: Define el modo de adquisición (en este caso, FINITE, que indica un número finito de muestras).
- numpy (np): Sierve para operaciones numéricas y generación de arreglos, como el eje de tiempo.
```python
fs = 1000                  
dur = 2 * 60                
n_samples = fs * dur      
```
- `fs`  Se establece la frecuencia de muestreo en 1000 Hz.
- `dur` es la duración de adquisición en segundos .
- `n_samples` Se calcula el número total de muestras a adquirir (1000 × 120 = 120,000 muestras).
---
1.2 Configuración y adquisición de datos
```python
with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev3/ai0")
    task.timing.cfg_samp_clk_timing(fs, sample_mode=AcquisitionType.FINITE, samps_per_chan=n_samples)
    task.start()
    task.wait_until_done(timeout=dur + 10)
    data = task.read(number_of_samples_per_channel=n_samples)
time_axis = np.linspace(0, duration_seconds, num_samples, endpoint=False)
```
En esta parte nos vasamos en el codigo presentado en [nidaqmx-python](https://github.com/ni/nidaqmx-python) el cual se modifico para que ampliara el tiempo de adquisisción de datos,donde:
- task.timing.cfg_samp_clk_timing(...):Configura la temporización de la adquisición:
  - fs: Frecuencia de muestreo.
  - sample_mode=AcquisitionType.FINITE: Se indica que la adquisición es de número finito de muestras.
  - samps_per_chan=n_samples: Número de muestras a adquirir por canal.
- task.start(): Inicia la adquisición de datos.
- data = task.read(...): Lee los datos adquiridos, esperando obtener el número total de muestras especificado.
---
1.3 Almacenamiento de Datos 
```python
with open("datos_adquiridos.txt", "w") as f:
    f.write("Tiempo (s)\tVoltaje (V)\n")
    for t, v in zip(time_axis, data):
        f.write(f"{t:.6f}\t{v:.6f}\n")
```
- with open("datos_adquiridos.txt", "w") as f: Abre (o crea) el archivo datos_adquiridos.txt en modo escritura.
- f.write("Tiempo (s)\tVoltaje (V)\n"): Escribe la línea de encabezado en el archivo, separando las columnas con una tabulación (\t).
- for t, v in zip(time_axis, data): Itera simultáneamente sobre el eje de tiempo (time_axis) y los datos de voltaje (data).
- f.write(f"{t:.6f}\t{v:.6f}\n"): Cada línea guarda un par tiempo-voltaje formateado a 6 decimales, separados por tabulación, en una línea individual.
---

### 2. Adquisición de la señal EMG.

<p align="center">
  <img src="https://github.com/SMD0927/Laboratorio-4/blob/main/Se%C3%B1aloriginal.jpg" alt="Señal EMG adquirida" width="850">
</p>

<p align="center">
  <img src="https://github.com/SMD0927/Laboratorio-4/blob/main/Se%C3%B1alOriginalFiltrada.jpg" alt="Señal EMG adquirida filtrada" width="850"> 
</p>

El primer diagrama muestra la señal electromiográfica captada para este laboratorio,es decir, sin implementarle ningun tipo de filtro. Observandose que presenta variaciones tanto de frencuencia como de amplitud indicando cambion en la activación muscular conforme avanza la prueba.
Debido a que esta es la señal original, se evidencia la existencia de ruido e interferencias, dificultando la interpretación precisa del comportamiento electromiográfico del musculo. La reducción gradual de la señal esta vinculada con la fatiga muscular, ya que se espera que conforme el esfuerzo se extienda, la capacidad contráctil se reduzca y esto se refleje en la actividad eléctrica registrada conforme al tiempo.

Por otro lado, en la segunda grafica muestra la misma señal electromiográfica que en el primero, pero utilizando un filtro de pasa-banda. Este filtro, fundamentado en la función bandpass_filter() del código suministrado, facilita la eliminación de frecuencias no deseadas, dejando solo las pertinentes para el análisis del EMG. Este filtrado facilita la identificación de patrones de activación muscular sin la interferencia de ruidos de baja frecuencia (como el movimiento del electrodo) o de alta frecuencia (como el sonido de la red eléctrica). En esta versión de la señal se puede apreciar con más exactitud el progreso de la fatiga muscular, dado que las oscilaciones electromiográficas están directamente vinculadas con la actividad de las unidades motoras.

```python
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut/nyquist, highcut/nyquist], btype='band')
    return lfilter(b, a, data)

filtered_voltage = bandpass_filter(voltage, lowcut, highcut, fs)

plt.figure(figsize=(20, 5))
plt.plot(t, filtered_voltage, label="Señal Filtrada", color='orange')
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal Filtrada")
plt.xlim(0,4)
plt.legend()
plt.show()
plt.grid()
```
<p align="center">
<img src="https://github.com/SMD0927/Laboratorio-4/blob/main/Se%C3%B1alOriginal10seg.jpg"alt="Sección de la Señal adquirida filtrada" width="850">
</p>

<p align="center">
    <img src="https://github.com/SMD0927/Laboratorio-4/blob/main/Se%C3%B1alOriginalFiltrada10seg.jpg" alt="Sección de la Señal adquirida" width="850">
</p>

Ahora se tiene la refresentación gráfica que en un primer caso muestra un segmento de los primeros 10 segundos de la señal inicial sin filtrado.  Este acercamiento se lleva a cabo para examinar el comportamiento inicial de la actividad muscular, anticipando un incremento en la intensidad de la señal a causa de la activación inicial de un mayor número de unidades motoras. A primera impresión, se puede apreciar que la señal conserva una amplitud considerable y una variabilidad estable, lo que indica una fuerte contracción muscular en los primeros segundos del esfuerzo.  No obstante, el hecho de que haya ruido en esta señal puede aún complicar el estudio exacto de la actividad electromiográfica.

Finalmente, la última gráfica muestra los primeros 10 segundos de la señal EMG con la aplicación del filtro pasa-banda mediante el mismo codigo de filtro para la señal original. Al igual que en la segunda gráfica, el filtrado mejora la calidad de la señal al eliminar interferencias y ruidos no deseados, permitiendo una mejor visualización de la activación muscular inicial. Se observa que, la señal filtrada exhibe una estructura más clara, facilitando la identificación de la frecuencia y la amplitud de la actividad electromiográfica en la fase inicial del esfuerzo. La comparación de esta gráfica con la anterior permite notar la importancia del filtrado para obtener una interpretación más precisa de la fatiga muscular.

---

### 3. Graficado de la Señal
```python
plt.figure(figsize=(10, 5))
plt.plot(señal, label="Señal fisiológica")
plt.title("ECG")
plt.xlabel("TIEMPO [ms]")
plt.ylabel("VOLTAJE [mV]")
plt.legend()
plt.grid()
plt.show()
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/d8104ccb-6b13-49c2-b510-abae7d5338f3" alt="image" width="500">
</p>

La gráfica muestra la señal ECG en función del tiempo, donde se pueden ver claramente las ondas características (P, QRS y T) y cómo varía el voltaje. Se aprecia un patrón cíclico que indica una actividad cardíaca regular, aunque también se observa algo de ruido en la línea base, lo que podría deberse a interferencias en la toma de datos.

---

### 4. Estadísticos Descriptivos
En esta sección se calculan estadísticas básicas de la señal de dos formas: de manera manual y usando NumPy. Ambas aproximaciones generan resultados muy similares: una media cercana a cero (-0.0124 o -0.012) lo que coincide con el histograma que revela una concentración de valores alrededor de este punto ,y una desviación estándar de 0.131, lo que indica que la señal está centrada y presenta una dispersión moderada. El coeficiente de variación, cercano a 10.55, lo que refleja una variabilidad relativa en la señal.[4]
#### 4.1. Cálculo Manual
```python
def estadisticos_programados():
    suma = 0
    for v in señal:
        suma += v    
    media = suma / t
    suma2 = sum((u - media)**2 for u in señal)
    desvesta = (suma2 / (t - 1))**0.5
    coeficiente = desvesta / abs(media)
    print('media:', media)
    print("desviacion estandar:", desvesta)
    print('coeficente de variacion', coeficiente)

estadisticos_programados()
```
Se calculan los siguientes estadísticos:
- **Media (μ):** Valor promedio de la señal.
- **Desviación Estándar (σ):** Medida de la dispersión de los datos respecto a la media.
- **Coeficiente de Variación (CV):** Relación entre desviación estándar y media, expresada en porcentaje.

$$
\mu = \frac{\sum x_i}{n}, \quad
\sigma = \sqrt{\frac{\sum (x_i - \mu)^2}{n-1}}, \quad
CV = \frac{\sigma}{\mu}
$$


**Resultados:**
- Media: -0.0124
- Desviación estándar: 0.131
- Coeficiente de variación: 10.557

**Interpretación:**
La media cercana a cero indica una señal centrada, mientras que el coeficiente de variación muestra una variabilidad moderada.

#### 4.2. Usando Funciones de NumPy
```python
def estadisticos_Bibliotecas():
    media = np.mean(señal)
    desvesta = np.std(señal, ddof=1)
    coeficiente = desvesta /abs(media)
    print('Media:', media)
    print("Desviación estándar:", desvesta)
    print('Coeficiente de variación:', coeficiente)

estadisticos_Bibliotecas()
```
Se obtienen los mismos resultados de manera más eficiente utilizando NumPy.

**Resultados:**
- Media: -0.012
- Desviación estándar: 0.131
- Coeficiente de variación: 10.554

---

### 5. Función de Probabilidad
```python
def calcular_funcion_probabilidad(senal):
    valores_unicos = np.unique(señal)
    probabilidades = {}
    for valor in valores_unicos:
        probabilidades[valor] = np.sum(señal == valor) / len(señal)
    for valor, prob in probabilidades.items():
        print(f"Valor: {valor:.5f}, Probabilidad: {prob:.5f}")

calcular_funcion_probabilidad(señal)
```
$$
P(v) = \frac{\text{Frecuencia Absoluta de } v}{\text{Total de Valores}}
$$

Se calcula la probabilidad de ocurrencia de cada valor único en la señal. Esto ayuda a comprender cómo se distribuyen los valores específicos.[5]

**Ejemplo de Resultados:**
- Valor: -0.28000, Probabilidad: 0.00050
- Valor: 0.00000, Probabilidad: 0.01650

**Análisis:**
La mayoría de los valores tienen baja probabilidad individual, lo que refleja la variabilidad natural de la señal.

---

### 6. Ruido Añadido y Cálculo de SNR
#### 6.1. Ruido Gaussiano
```python
ruido = np.random.normal(0, 0.04, t) 
señal_ruidosa = señal + ruido 
```
El ruido gaussiano es un tipo de ruido aleatorio cuyas variaciones siguen una distribución normal.[6] Se define por su media (0 en este caso) y su desviación estándar (0.1, que controla su intensidad). Es común en señales fisiológicas debido a la electrónica del sistema de adquisición y otras fuentes de interferencia aleatoria.

#### 6.1. Ruido Gaussiano Amplificado
```python
ruido4 = np.random.normal(0, 0.1, t) 
señal_ruidosa = señal + ruido4 
```

#### 6.2. Ruido de Impulso
```python
prob_impulso = 0.08
impulsos = np.random.choice([0, 1], size=len(señal), p=[1-prob_impulso, prob_impulso])
amplitud_impulso = np.random.choice([-1, 1], size=len(señal)) * 0.2
ruido2 = impulsos * amplitud_impulso
```
Este ruido se caracteriza por picos abrupto y repentinos en la señal [7], generados aquí con una probabilidad del 8% (prob_impulso = 0.08). La función np.random.choice determina en qué puntos aparecen los impulsos (1 o 0), y la amplitud se asigna aleatoriamente con valores de ±0.2. Este ruido suele deberse a interferencias externas o fallos en la transmisión de datos.

#### 6.2. Ruido de Impulso Amplificado
```python
prob = 0.08
im = np.random.choice([0, 1], size=len(señal), p=[1-prob, prob])
am = np.random.choice([-1, 1], size=len(señal)) * 0.4
ruido5 = im * am
```
#### 6.3. Ruido Tipo Artefacto 
```python
prob_imp = 0.15
impul = np.random.choice([0, 1], size=len(señal), p=[1-prob_imp, prob_imp])
amplitud = np.random.choice([-1, 1], size=len(señal)) * 0.2
ruido3 = impul * amplitud
```
Este ruido representa alteraciones no deseadas en la señal, que no se encuentran presentes en la fuente original si no que se deben a alteraciones externas a dicha fuente[8], como movimientos del paciente o fallos en los electrodos. Es similar al ruido de impulso, pero con una mayor probabilidad de ocurrencia (prob_imp = 0.15). Se genera con la misma lógica de np.random.choice, agregando perturbaciones aleatorias.

#### 6.3. Ruido Tipo Artefacto Amplificado
```python
p = 0.2
i = np.random.choice([0, 1], size=len(señal), p=[1-p, p])
a = np.random.choice([-1, 1], size=len(señal)) * 0.4
ruido6 = i * a
```
#### Cálculo del SNR
El SNR (Relación Señal-Ruido) cuantifica qué tan fuerte es la señal en comparación con el ruido presente. Un valor alto indica una señal clara con poca interferencia, mientras que un valor bajo implica que el ruido domina, dificultando su interpretación [9]. Se calcula como:

$$
\text{SNR (dB)} = 10 \cdot \log_{10} \left( \frac{P_{\text{señal}}}{P_{\text{ruido}}} \right)
$$

```python
def snr(s,r):
    potencia_señal = np.mean(s**2)
    potencia_ruido = np.mean(r**2)
    
    if potencia_ruido == 0:
        return np.inf
    snr = 10 * np.log10(potencia_señal/potencia_ruido) 
    return snr
```
**Señales con Ruido:**
<p align="center">
    <img src="https://github.com/user-attachments/assets/1f1d2a8e-0e72-49a7-9e39-701a1fda1e9f" alt="image" width="500">
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/776a5151-430b-4e1d-b91e-a53ff1d90979" alt="image" width="500">
</p>
<p align="center">
    <img src="https://github.com/user-attachments/assets/6ed6d849-933b-41c5-8693-b9a34e554b67" alt="image" width="500">
</p>

**Señales con Ruido Amplificado:**
<p align="center">
    <img src="https://github.com/user-attachments/assets/924d3f6d-c0eb-4e05-a678-a451cb81b9d4" alt="image" width="500">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/1183d8bd-2dd4-40ef-93bb-302e7420e4d9" alt="image" width="500">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/080b90b3-ae34-447e-b9be-ac5e3a59199f" alt="image" width="500">
</p>





**Resultados SNR:**
- **Ruido Gaussiano:** 10.419 dB  
- **Ruido Impulso:** 7.274 dB
- **Ruido Artefacto:** 4.066 dB
- **Ruido Gaussiano Amplificado:** 2.126 dB  
- **Ruido Impulso Amplificado:** 1.336 dB
- **Ruido Artefacto Amplificado:** -2.51 dB

**Análisis:**
Las gráficas muestran que al aumentar la amplitud del ruido, el SNR disminuye, lo que significa que la señal se vuelve menos distinguible. Con una amplitud baja de ruido, la señal sigue siendo reconocible y el SNR es mayor. Sin embargo, al amplificar el ruido, el SNR cae drásticamente, generando una señal más contaminada y difícil de interpretar. Esto refleja la importancia de minimizar el ruido en aplicaciones donde la precisión es fundamental, como en el procesamiento de señales fisiológicas.


---

### 7. Visualización de Ruido
Se grafican las señales contaminadas con ruido:
```python
plt.figure()
plt.plot(señal, label='Señal original')
plt.plot(señal_ruidosa, label='Ruido gaussiano')
plt.legend()
plt.show()
```

**Análisis Visual:**
El código grafica la señal original junto con su versión afectada por ruido gaussiano, permitiendo visualizar cómo este ruido altera la forma de la señal.
Las gráficas muestran cómo diferentes tipos de ruido afectan la forma de la señal, con el ruido de artefacto generando mayores distorsiones.

---

## Instrucciones
1. Captar la señal electromiografica usando el dispositivo de adquisición de datos(DAQ).
2. Codificar y Ejecutar el código en un entorno Python.
---

## Bibliografías

[1] https://doi.org/10.4321/S1137-66272009000600003

[2] https://nidaqmx-python.readthedocs.io/en/stable/

----
## Autores 
Sañuel Peña -Ana Abril- Santiago Mora

