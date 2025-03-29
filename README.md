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
- `task.timing.cfg_samp_clk_timing(...)`:Configura la temporización de la adquisición:
  - fs: Frecuencia de muestreo.
  - sample_mode=AcquisitionType.FINITE: Se indica que la adquisición es de número finito de muestras.
  - samps_per_chan=n_samples: Número de muestras a adquirir por canal.
- `task.start()`: Inicia la adquisición de datos.
- `data = task.read(...)`: Lee los datos adquiridos, esperando obtener el número total de muestras especificado.
---
1.3 Almacenamiento de Datos 
```python
with open("datos_adquiridos.txt", "w") as f:
    f.write("Tiempo (s)\tVoltaje (V)\n")
    for t, v in zip(time_axis, data):
        f.write(f"{t:.6f}\t{v:.6f}\n")
```
- `with open("datos_adquiridos.txt", "w") as f:` Abre (o crea) el archivo datos_adquiridos.txt en modo escritura.
- `f.write("Tiempo (s)\tVoltaje (V)\n")`: Escribe la línea de encabezado en el archivo, separando las columnas con una tabulación (\t).
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

### 3. Segmentación y ventaneo de la señal EMG
Para medir la fatiga muscular, se requiere comparar la actividad muscular al comienzo y al término del experimento.  Por esta razón, la señal filtrada se segmenta en ventanas concretas:

1. Primeras contracciones (sin fatiga): Se extraen las primeras 4.8 s de la señal.

2. Últimas contracciones (con fatiga): Se extraen ventanas en los últimos segundos.

Para optimizar la evaluación en el ámbito de la frecuencia, se utiliza una ventana de Hann, la cual disminuye los impactos de discontinuidad en los bordes de la señal mediante el cálculo de su transformada Fourier.

```python
hanning = np.hanning(1000) 
ventana1 = sf[:1000] * hanning
ventana2 = sf[1000:2000] * hanning
ventana3 = sf[2000:2700] * hanning[:700]
```
Es importante realizar este ventaneo ya que minimiza las fugas espectrales en la FFT, incrementando la exactitud del análisis en el campo de la frecuencia. Asimismo, permite comparar las ventanas de la señal, lo que posibilita valorar el efecto de la fatiga muscular.


---
### 4. Dominio de la Frecuencia
En el análisis de la fatiga muscular, uno de los indicadores esenciales es la frecuencia mediana (FM) de la señal EMG. Para determinar su valor, se efectúa una Transformada Rápida de Fourier (FFT) en cada ventana de la señal.
```python
fre = np.fft.fftfreq(N, 1/fs)
frecuencias = fre[:N//2]
espectro = np.fft.fft(ventana) / N
magnitud = 2 * np.abs(espectro[:N//2])
```
Este espectro nos proporciona la frecuencia mediana (FM), que se define como la frecuencia en la que se concentra el 50% de la energía total de la señal:
```python
psd = magnitud ** 2
potencia_total = np.sum(psd)
potencia_acumulada = np.cumsum(psd)
fm_index = np.where(potencia_acumulada >= potencia_total / 2)[0][0]
fm = frecuencias[fm_index]
```
Este análisis es fundamental, ya que una disminución de la FM a lo largo del tiempo es indicativa de fatiga muscular. Es necesario indicar que se grafican los espectros de cada ventana para visualizar la evolución de la señal EMG en el dominio de la frecuencia.

#### 5. Estadística de la Fatiga Muscular

Para determinar si la fatiga muscular afecta significativamente la frecuencia mediana, se emplea una prueba t pareada (ttest_rel) que compara las FM de las primeras y últimas contracciones.
```python
stat, p = ttest_rel(antes, despues)

if p < 0.05:
    print("Rechazamos la hipótesis H₀, la fatiga afecta significativamente la mediana de frecuencia.")
else:
    print("No se puede rechazar H₀, no hay evidencia suficiente de un cambio significativo.")
```
La interpretacion de estos resultados se expresa de la siguiente forma dado el caso:

Si \( p < 0.05 \): Se rechaza \( H0 \), lo que indica que la fatiga muscular ha causado una disminución significativa en la frecuencia mediana.

Si \( p &ge; 0.05 \): No hay suficiente evidencia para rechazar \( H0 \), lo que sugiere que la fatiga no ha afectado significativamente la FM.

**Análisis:**
Las gráficas muestran que al aumentar la amplitud del ruido, el SNR disminuye, lo que significa que la señal se vuelve menos distinguible. Con una amplitud baja de ruido, la señal sigue siendo reconocible y el SNR es mayor. Sin embargo, al amplificar el ruido, el SNR cae drásticamente, generando una señal más contaminada y difícil de interpretar. Esto refleja la importancia de minimizar el ruido en aplicaciones donde la precisión es fundamental, como en el procesamiento de señales fisiológicas.


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

