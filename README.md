# Laboratorio-4
Analisis de electromiografia (EMG)
# Análisis Estadístico de la Señal

En este laboratorio exploramos señales fisiológicas de ECG utilizando técnicas de estadística descriptiva y modelos de ruido. El objetivo es entender tanto las características propias de la señal como el impacto del ruido, analizando aspectos como la relación señal-ruido (SNR).

## Requisitos
- **Python 3.9**
- Bibliotecas necesarias:
  - `wfdb`
  - `numpy`
  - `matplotlib`
  - `seaborn`

Instalar dependencias:
```bash
pip install wfdb numpy matplotlib seaborn
```

## Estructura del Código

### 1. Lectura de Datos
```python
import wfdb
import numpy as np

datos = wfdb.rdrecord('rec_2')
t = 2000
señal = datos.p_signal[:t, 0]
```
Se utiliza `wfdb.rdrecord` para cargar una señal fisiológica (ECG) desde un archivo estándar en formato WFDB que fueron descargados en PhysioNet. En este caso, se seleccionan los primeros 2000 puntos de la señal. Este paso inicial permite trabajar con un subconjunto significativo de datos para realizar análisis detallados.[1][2]

---

### 2. Histograma de la Señal
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(señal, kde=True, bins=30, color='red')
plt.hist(señal, bins=30, edgecolor='blue')
plt.title('Histograma de Datos')
plt.xlabel('Datos')
plt.ylabel('Frecuencia')
plt.show()
```
<p align="center">
    <img src="https://i.postimg.cc/50qyPvY9/histograma.png" alt="histograma" width="450">
</p>

El histograma muestra una distribución asimétrica con mayor concentración de valores cerca de 0 y una cola extendida a la derecha, indicando un sesgo positivo. Esto sugiere la posible presencia de ruido o eventos atípicos en la señal, aunque la mayoría de los valores se mantienen dentro de un rango fisiológico típico.[3]

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
1. Descargar la señal desde bases de datos como PhysioNet.
2. Codificar y Ejecutar el código en un entorno Python.
---

## Bibliografías
[1] https://physionet.org/about/database/ 

[2] https://wfdb.readthedocs.io/en/latest/

[3] https://acortar.link/8Ua7sO

[4] http://ri.uaemex.mx/oca/view/20.500.11799/32031/1/secme-21225.pdf

[5] http://www.liceobrainstorm.cl/wp-content/uploads/2020/05/3ro-y-4to-medio-Electivo-de-Probabilidad-PPT-n%C2%B0-1-04-al-08-de-Mayo.pdf

[6] https://es.statisticseasily.com/glossario/what-is-gaussian-noise/

[7] https://svantek.com/es/servicios/ruido-de-impulso/

[8] https://www.uned.es/universidad/facultades/dam/jcr:aec2c175-f79e-4478-a0ed-ffec97816b5d/PFM_%20Luis_Alberto_Ramon_Surutusa.pdf

[9] https://wraycastle.com/es/blogs/knowledge-base/what-does-snr-stand-for?srsltid=AfmBOor-3cFfdYqIcESTfUynthEkfkz5Uz297oMsF_l-1v_Kda3J1Us_

----
## Autores 
Sañuel Peña -Ana Abril- Santiago Mora

