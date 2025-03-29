import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.stats import ttest_rel


señal = np.loadtxt("datos_adquiridos.txt", skiprows=1)        
tiempo = señal[:,0]  
voltaje = señal[:,1]

plt.figure(figsize=(10, 5))
plt.plot(tiempo, voltaje,color="b", label="Señal")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Gráfica de la Señal Adquirida Completa")
plt.grid()
fs = 1000

def filtro(s,fs):
    orden = 10
    corte1 = 400
    corte2 = 10
    nyquist = 0.5 * fs
    corte_normalizada1 = corte1 / nyquist
    corte_normalizada2 = corte2 / nyquist
    b, a = butter(orden, corte_normalizada1, btype='low', analog=False)
    b2, a2 = butter(orden, corte_normalizada2, btype='high', analog=False)
    señal_f1 = lfilter(b, a, s)
    return lfilter(b2, a2, señal_f1)    
sf = filtro(voltaje,fs)


# Ventaneo primeras contracciones
hanning = np.hanning(1000) 
ventana1 = sf[:1000] * hanning
ventana2 = sf[1000:2000] * hanning
ventana3 = sf[2000:2700] * hanning[:700]
ventana4 = sf[2700:3400] * hanning[:700]
ventana5 = sf[3400:4100] * hanning[:700]
ventana6 = sf[4100:4800] * hanning[:700]
señal_ventaneada = np.concatenate([ventana1, ventana2, ventana3, ventana4, ventana5, ventana6])
            
plt.figure(figsize=(18, 10))
        
plt.subplot(2, 1, 1)
plt.plot(tiempo[:4800], sf[:4800], color="b", label="Señal Original")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal Filtrada sin Ventana (Primeras contracciones)")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(tiempo[:4800], señal_ventaneada, color="r", label="Señal Ventaneada")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal filtrada con ventana (Primeras contracciones)")
plt.grid()

#ventaneo ultimas contracciones
ventana7 = sf[82400:83000] * hanning[:600]
ventana8 = sf[83000:83660] * hanning[:660]
ventana9 = sf[83660:84560] * hanning[:900]
ventana10 = sf[84560:85100] * hanning[:540]
ventana11 = sf[85100:85570] * hanning[:470]
ventana12 = sf[85570:86400] * hanning[:830]
señal_ventaneadaf = np.concatenate([ventana7, ventana8, ventana9, ventana10, ventana11, ventana12])
            
plt.figure(figsize=(18, 10))
        
plt.subplot(2, 1, 1)
plt.plot(tiempo[82400:86400], sf[82400:86400], color="b")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal Filtrada sin Ventana (ultimas contracciones)")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(tiempo[82400:86400], señal_ventaneadaf, color="r")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal filtrada con ventana (ultimas contracciones)")
plt.grid()

antes = []
despues = []        

for i in range(1, 13):
    ventana = eval(f"ventana{i}")  
    N = len(ventana)
    
    fre = np.fft.fftfreq(N, 1/fs)
    frecuencias = fre[:N//2]
    espectro = np.fft.fft(ventana) / N
    magnitud = 2 * np.abs(espectro[:N//2])
    
    psd = magnitud ** 2
    potencia_total = np.sum(psd)
    potencia_acumulada = np.cumsum(psd)
    
    fm_index = np.where(potencia_acumulada >= potencia_total / 2)[0][0]
    fm = frecuencias[fm_index]

    plt.figure(figsize=(10, 5))
    plt.plot(frecuencias, magnitud, color="k", label="Señal")
    plt.axvline(fm, color="r", linestyle="--", label=f"FM = {fm:.2f} Hz")
    plt.xlim(0, 700)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.title(f"Espectro de la Señal (ventana {i})")
    plt.legend()
    plt.grid()
    
    print(f"Frecuencia Mediana de la ventana {i}: {fm:.2f} Hz")
    
    if 1<=i<=6:
        antes.append(fm)
    else:
        despues.append(fm)
 
print()
stat, p = ttest_rel(antes, despues)

print(f'P-valor: {p}')

if p < 0.05:
    print("Rechazamos la hipotesis H₀, La fatiga afecta significativamente la mediana de frecuencia.")
else:
    print("No se puede rechazar H₀, No hay evidencia suficiente de un cambio significativo.")


plt.show()
