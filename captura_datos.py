import nidaqmx
from nidaqmx.constants import AcquisitionType
import numpy as np

sample_rate = 1000         
duration_minutes = 2      
duration_seconds = duration_minutes * 60  
num_samples = int(sample_rate * duration_seconds)  

with nidaqmx.Task() as task:
    
    task.ai_channels.add_ai_voltage_chan("Dev3/ai0")
    
    task.timing.cfg_samp_clk_timing(
        sample_rate,
        sample_mode=AcquisitionType.FINITE,
        samps_per_chan=num_samples
    )
    
    task.start()

    task.wait_until_done(timeout=duration_seconds + 10)

    data = task.read(number_of_samples_per_channel=num_samples)

time_axis = np.linspace(0, duration_seconds, num_samples, endpoint=False)
with open("datos_adquiridos.txt", "w") as archivo_txt:
    archivo_txt.write("Tiempo (s)\tVoltaje (V)\n")
    for t, v in zip(time_axis, data):
        archivo_txt.write(f"{t:.6f}\t{v:.6f}\n")
