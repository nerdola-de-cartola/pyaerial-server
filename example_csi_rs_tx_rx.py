#!/usr/bin/env python
# coding: utf-8

# # Using pyAerial to run CSI-RS transmission and reception
# 
# The pyAerial library supports transmission of 5G NR compliant CSI reference signals (CSI-RS), as well as the UE side channel estimation based on CSI-RS. This example shows how to use the pyAerial cuPHY Python bindings to run CSI-RS transmission and reception using the pyAerial CSI-RS transmitter and receiver pipelines. The notebook runs CSI-RS transmission and reception and plots the estimated channel against the actual channel realization, for the given signal-to-noise ratio.
# 
# The NVIDIA [Sionna](https://nvlabs.github.io/sionna/) library is utilized for simulating the radio channel based on 3GPP channel models.

# ## Imports

# In[3]:


#get_ipython().run_line_magic('matplotlib', 'widget')
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Silence TensorFlow.

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import sionna
import tensorflow as tf

# pyAerial imports
from aerial.phy5g.csirs import CsiRsConfig
from aerial.phy5g.csirs import CsiRsTxConfig
from aerial.phy5g.csirs import CsiRsRxConfig
from aerial.phy5g.csirs import CsiRsTx
from aerial.phy5g.csirs import CsiRsRx
from aerial.util.cuda import get_cuda_stream

# Configure the notebook to use only a single GPU and allocate only as much memory as needed.
# For more details, see https://www.tensorflow.org/guide/gpu.
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# ## Parameters
# Set channel and numerology parameters.

# In[4]:


# Channel parameters
esno_db = 20.              # Es/No
num_tx_ant = 8             # Number of Tx antennas
num_rx_ant = 2             # Number of Rx antennas
carrier_frequency = 3.5e9  # Carrier frequency in Hz
delay_spread = 100e-9      # Nominal delay spread in [s]. Please see the CDL documentation
                           # about how to choose this value.
link_direction = "downlink"
channel_model = "CDL-D"    # Channel model: Suitable values:
                           # "Rayleigh" - Rayleigh block fading channel model (sionna.channel.RayleighBlockFading)
                           # "CDL-x", where x is one of ["A", "B", "C", "D", "E"] - for 3GPP CDL channel models
                           #          as per TR 38.901.
speed = 0.8333             # UE speed [m/s].

# Numerology and frame structure. See 3GPP TS 38.211.
num_symb_per_slot = 14
fft_size = 4096
subcarrier_spacing = 30e3
num_symbols = 14
num_slots_per_frame = 20
num_prb = 273


# ## CSI-RS configuration
# Set CSI-RS resource mapping and sequence configuration. Refer to 3GPP TS 38.211 section 7.4.1.5.3 and in particular table 7.4.1.5.3-1 for the exact definitions of the fields. The parameterization here follows closely the 3GPP specification.
# 
# Note: CSI-RS type is by default non-zero power (NZP) CSI-RS - only this is currently supported by cuPHY.

# In[5]:


csirs_configs = [CsiRsConfig(
    start_prb=0,              # Start PRB.
    num_prb=num_prb,          # Number of PRBs.
    freq_alloc=[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Bitmap defining the frequencyDomainAllocation field in CSI-RS RRC parameters.
    row=6,                    # CSI-RS parameter table row number.
    symb_L0=0,                # Time domain location L0. This corresponds to the `firstOFDMSymbolInTimeDomain`
                              # field in CSI-RS RRC parameters.
    symb_L1=8,                # Time domain location L1. This corresponds to the
                              # `firstOFDMSymbolInTimeDomain2` field in CSI-RS RRC parameters.
    freq_density=2,           # The `density` field in CSI-RS RRC parameters.
    scramb_id=0,              # CSI-RS scrambling ID.
    idx_slot_in_frame=0,      # Slot index in frame.
    cdm_type=1,               # CDM Type - this is the `cdm-Type` in CSI-RS RRC parameters.
    beta=1.0                  # CSI-RS power control.
)]


# ## Create CSI-RS transmitter and receiver objects
# 
# The CSI-RS transmitter and receiver objects are created here. Also, the dynamically changing slot configurations for both the transmitter and the receiver are instantiated. These are given as parameters later when the Tx/Rx are actually called.

# In[6]:


cuda_stream = get_cuda_stream()

csirs_rx_config = CsiRsRxConfig(
    csirs_configs=[csirs_configs], # One cell with multiple CSI-RS configurations.
    ue_cell_association=[0]        # One UE associated with cell 0.
)

csirs_tx_config = CsiRsTxConfig(
    csirs_configs=[csirs_configs],  # One cell with multiple CSI-RS configurations.
    precoding_matrices=[]           # No precoding.
)

csirs_rx = CsiRsRx(num_prb_dl_bwp=[273], cuda_stream=cuda_stream)
csirs_tx = CsiRsTx(num_prb_dl_bwp=[273], cuda_stream=cuda_stream)


# ## Channel generation using Sionna
# Simulating the transmission through the radio channel takes advantage of the channel model implementations available in NVIDIA Sionna. In Sionna, the transmission can be simulated directly in frequency domain by defining a resource grid. In our case, reference signal patterns and data carrying resource elements are defined elsewhere within the Aerial code, hence we define resource grid as a simple dummy grid containing only data symbols. Also, instead of the actual FFT size we just pass the number of used subcarriers - everything gets run in frequency domain.
# 
# See also: [Sionna documentation](https://nvlabs.github.io/sionna/index.html)

# In[7]:


# Define the resource grid.
resource_grid = sionna.phy.ofdm.ResourceGrid(
    num_ofdm_symbols=num_symb_per_slot,
    fft_size=273*12,
    subcarrier_spacing=subcarrier_spacing,
)

# Define the antenna arrays.
ue_array = sionna.phy.channel.tr38901.AntennaArray(
    num_rows=1,
    num_cols=int(num_rx_ant/2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency
)
gnb_array = sionna.phy.channel.tr38901.AntennaArray(
    num_rows=1,
    num_cols=int(num_tx_ant/2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency
)

if channel_model == "Rayleigh":
    ch_model = sionna.phy.channel.RayleighBlockFading(
        num_rx=1,
        num_rx_ant=num_rx_ant,
        num_tx=1,
        num_tx_ant=num_tx_ant
    )

elif "CDL" in channel_model:
    cdl_model = channel_model[-1]

    # Configure a channel impulse reponse (CIR) generator for the CDL model.
    ch_model = sionna.phy.channel.tr38901.CDL(
        cdl_model,
        delay_spread,
        carrier_frequency,
        ue_array,
        gnb_array,
        link_direction,
        min_speed=speed
    )
else:
    raise ValueError(f"Invalid channel model {channel_model}!")

ofdm_channel = sionna.phy.channel.OFDMChannel(
    ch_model,
    resource_grid,
    add_awgn=True,
    normalize_channel=True,
    return_channel=True
)

def apply_channel(tx_tensor, No):
    # We use DLPack to keep the tensors on the GPU between pyAerial and Sionna.
    tx_tensor = tf.experimental.dlpack.from_dlpack(cp.ascontiguousarray(tx_tensor).toDlpack())

    # Add batch and num_tx dimensions that Sionna expects.
    tx_tensor = tf.transpose(tx_tensor, (2, 1, 0))[None, None]
    rx_tensor, channel = ofdm_channel(tx_tensor, No)   
    rx_tensor = tf.transpose(rx_tensor[0, 0], (2, 1, 0))  # Remove them.
    channel = tf.transpose(channel[0, 0, :, 0, ...], (3, 2, 1, 0))
    
    rx_tensor = tf.experimental.dlpack.to_dlpack(rx_tensor)
    rx_tensor = cp.from_dlpack(rx_tensor)
    return rx_tensor, channel


# ## Run the CSI-RS transmission and reception
# Run the CSI-RS generation at the transmitter side, pass the frequency-domain slot signal through the radio channel, and run the UE side CSI-RS channel estimation.

# In[8]:


No = pow(10., -esno_db / 10.)

tx_buffer = cp.zeros((num_prb * 12, num_symb_per_slot, num_tx_ant), dtype=cp.complex64)
tx_buffer = csirs_tx(config=csirs_tx_config, tx_buffers=[tx_buffer])[0]

rx_data, channel = apply_channel(tx_buffer, No)
channel = channel.numpy()

ch_est = csirs_rx(rx_data=[rx_data], config=csirs_rx_config)
ch_est = ch_est[0][0].get()


# ## Plot channel estimation results

# In[9]:


# Number of first PRBs to plot (for better visualization)
num_prb_to_plot = 20

subc_idx = np.arange(0, num_prb_to_plot * 12, 12)
for tx_ant in range(num_tx_ant):
    for rx_ant in range(num_rx_ant):    
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"CSI-RS channel estimates for Tx antenna {tx_ant} / Rx antenna {rx_ant}")    
        axs[0].plot(np.real(ch_est[:num_prb_to_plot, tx_ant, rx_ant]), 'bo', label='Channel estimates')
        axs[0].plot(np.real(channel[subc_idx, 0, tx_ant, rx_ant]), 'k:', label='Channel')
        axs[1].plot(np.imag(ch_est[:num_prb_to_plot, tx_ant, rx_ant]), 'bo', label='Channel estimates')
        axs[1].plot(np.imag(channel[subc_idx, 0, tx_ant, rx_ant]), 'k:', label='Channel')                   
        axs[0].set_title("Real part")
        axs[1].set_title("Imaginary part")
        axs[0].set_ylim(np.real(ch_est[:num_prb_to_plot, ...]).min(), np.real(ch_est[:num_prb_to_plot, ...]).max())
        axs[1].set_ylim(np.imag(ch_est[:num_prb_to_plot, ...]).min(), np.real(ch_est[:num_prb_to_plot, ...]).max())
        for ax in axs:
            ax.grid(True)
            ax.set_xlim(0, num_prb_to_plot)
            ax.set_xlabel('PRB index')
            ax.legend()
        axs[0].grid(True)
        axs[1].grid(True)
        plt.show()


# In[ ]:




