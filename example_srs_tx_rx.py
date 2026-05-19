#!/usr/bin/env python
# coding: utf-8

# # Using pyAerial to run 5G sounding reference signal transmission and reception
# This example shows how to use the pyAerial cuPHY Python bindings to run sounding reference signal (SRS) transmission and reception using the pyAerial SRS transmitter and receiver pipelines.
# 
# The NVIDIA [Sionna](https://nvlabs.github.io/sionna/) library is utilized for simulating the radio channel based on 3GPP channel models.

# ## Imports

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'widget')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Silence TensorFlow.

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import sionna
import tensorflow as tf

from aerial.phy5g.srs import SrsTx
from aerial.phy5g.srs import SrsRx
from aerial.phy5g.srs import SrsTxConfig
from aerial.phy5g.srs import SrsRxConfig
from aerial.phy5g.srs import SrsConfig
from aerial.phy5g.srs import SrsRxUeConfig
from aerial.phy5g.srs import SrsRxCellConfig
from aerial.util.cuda import get_cuda_stream

# Configure the notebook to use only a single GPU and allocate only as much memory as needed.
# For more details, see https://www.tensorflow.org/guide/gpu.
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# ## Simulation parameters
# 

# In[2]:


esno_db = 40

# Numerology and frame structure. See TS 38.211.
num_symb_per_slot = 14
fft_size = 4096
cyclic_prefix_length = 288
subcarrier_spacing = 30e3
num_guard_subcarriers = (410, 410)
num_slots_per_frame = 20
srs_symbols = [13]

# Channel parameters
num_ue_tx_ant = 1
num_gnb_rx_ant = 4
carrier_frequency = 3.5e9  # Carrier frequency in Hz.
delay_spread = 100e-9      # Nominal delay spread in [s]. Please see the CDL documentation
                           # about how to choose this value.
link_direction = "uplink"
channel_model = "CDL-D"    # Channel model: Suitable values:
                           # "Rayleigh" - Rayleigh block fading channel model (sionna.channel.RayleighBlockFading)
                           # "CDL-x", where x is one of ["A", "B", "C", "D", "E"] - for 3GPP CDL channel models
                           #          as per TR 38.901.
speed = 0.8333             # UE speed [m/s]. The direction of travel will chosen randomly within the x-y plane.


# ## Build the pipelines
# We build the SRS transmitter and receiver pipeline objects here.

# In[3]:


# Generate a CUDA stream for running the whole thing.
cuda_stream = get_cuda_stream()

srs_tx = SrsTx(
    num_max_srs_ues=1,                  # Maximum number of UEs for which this object will handle SRS Tx. Here just one.
    num_slot_per_frame=num_slots_per_frame,
    num_symb_per_slot=num_symb_per_slot,
    cuda_stream=cuda_stream
)

srs_rx = SrsRx(
    num_rx_ant=[num_gnb_rx_ant],        # A list, one element per cell.
    chest_algo_idx=0,                   # MMSE.
    enable_delay_offset_correction=1,
    chest_params=None,                  # Use defaults.
    num_max_srs_ues=1,                  # Maximum number of UEs for which this object will handle SRS Rx.
    cuda_stream=cuda_stream
)


# ## Sounding reference signal and SRS Tx and Rx pipeline slot configurations
# Define the SRS signal configuration for the slot, as well as the slot configurations for the SRS Tx and Rx pipelines built above. These are the dynamic configurations that depend on slot and frame index, and other parameters.
# 
# The SRS signal parameters follow essentially the 3GPP specifications, 3GPP TS 38.211 in particular.

# In[4]:


# Slot and frame indices.
frame = 0
slot = 0

# SRS signal configuration. Just one UE.
srs_config = SrsConfig(
    num_ant_ports=1,
    num_syms=len(srs_symbols),
    num_repetitions=1,
    comb_size=2,
    start_sym=srs_symbols[0],
    sequence_id=0,
    config_idx=63,
    bandwidth_idx=0,
    comb_offset=0,
    cyclic_shift=0,
    frequency_position=0,
    frequency_shift=0,
    frequency_hopping=0,
    resource_type=0,
    periodicity=1,
    offset=0,
    group_or_sequence_hopping=0
)

# UE SRS transmitter pipeline slot configuration. One UE, one SRS signal configuration.
srs_tx_config = SrsTxConfig(
    slot=slot,
    frame=frame,
    srs_configs=[srs_config]
)

# gNB SRS receiver pipeline slot configuration
# - UEs from which SRS are received
srs_rx_ue_config = SrsRxUeConfig(
    cell_idx=0,                                             # Cell association.
    srs_config=srs_config,                                  # SRS signal configuration.
    srs_ant_port_to_ue_ant_map=np.zeros(1, dtype=np.uint8), # Mapping to UE antennas.
    prg_size=2,                                             # PRB group size.
    start_prg=0,                                            # Start PRB group.
    num_prgs=136                                            # 273 // prg_size.
)

# - Cells handled by this pipeline.
srs_rx_cell_config = SrsRxCellConfig(
    slot=slot,
    frame=frame,
    srs_start_sym=srs_symbols[0],
    num_srs_sym=len(srs_symbols)
)

# - The actual slot configuration.
srs_rx_config = SrsRxConfig(
    srs_cell_configs=[srs_rx_cell_config],
    srs_ue_configs=[srs_rx_ue_config]
)


# ## Channel generation using Sionna
# Simulating the transmission through the radio channel takes advantage of the channel model implementations available in NVIDIA Sionna. In Sionna, the transmission can be simulated directly in frequency domain by defining a resource grid. We define the resource grid as a simple dummy grid containing only data symbols as the SRS are defined elsewhere in the code.
# 
# See also: [Sionna documentation](https://nvlabs.github.io/sionna/index.html)

# In[5]:


# Define the resource grid.
resource_grid = sionna.phy.ofdm.ResourceGrid(
    num_ofdm_symbols=len(srs_symbols),  # Simulate just the SRS symbols.
    fft_size=fft_size,
    subcarrier_spacing=subcarrier_spacing,
    num_tx=num_ue_tx_ant,
    num_streams_per_tx=1,
    cyclic_prefix_length=cyclic_prefix_length,
    num_guard_carriers=num_guard_subcarriers,
    dc_null=False,
    pilot_pattern=None,
    pilot_ofdm_symbol_indices=None
)
resource_grid_mapper = sionna.phy.ofdm.ResourceGridMapper(resource_grid)
remove_guard_subcarriers = sionna.phy.ofdm.RemoveNulledSubcarriers(resource_grid)

# Define the antenna arrays.
ue_array = sionna.phy.channel.tr38901.Antenna(
    polarization="single",
    polarization_type="V",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency
)
gnb_array = sionna.phy.channel.tr38901.AntennaArray(
    num_rows=1,
    num_cols=int(num_gnb_rx_ant/2),
    polarization="dual",
    polarization_type="cross",
    antenna_pattern="38.901",
    carrier_frequency=carrier_frequency
)

if channel_model == "Rayleigh":
    ch_model = sionna.phy.channel.RayleighBlockFading(
        num_rx=1,
        num_rx_ant=num_gnb_rx_ant,
        num_tx=1,
        num_tx_ant=num_ue_tx_ant
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
    """Transmit the Tx tensor through the radio channel."""
    # We use DLPack to keep the tensors on the GPU between pyAerial and Sionna.
    tx_tensor = tf.experimental.dlpack.from_dlpack(cp.ascontiguousarray(tx_tensor).toDlpack())

    # Add batch and num_tx dimensions that Sionna expects and reshape.    
    tx_tensor = tf.transpose(tx_tensor, (2, 1, 0))
    tx_tensor = tf.reshape(tx_tensor, (1, -1))[None, None]    
    tx_tensor = resource_grid_mapper(tx_tensor)
    rx_tensor, channel = ofdm_channel(tx_tensor, No)
    rx_tensor = remove_guard_subcarriers(rx_tensor)
    channel = remove_guard_subcarriers(channel)
    rx_tensor = rx_tensor[0, 0]
    channel = tf.transpose(channel[0, 0, :, 0, 0, :, :], (2, 1, 0))
    rx_tensor = tf.transpose(rx_tensor, (2, 1, 0))

    rx_tensor = tf.experimental.dlpack.to_dlpack(rx_tensor)
    rx_tensor = cp.from_dlpack(rx_tensor)    
    return rx_tensor, channel


# ## Run the SRS transmission and reception
# We run the SRS transmitter, using the transmitter configuration as an argument. Then the generated SRS signal gets transmitted through the radio channel (using the Sionna library here). The received tensor is fed into the SRS receiver pipeline.

# In[6]:


# Take the Tx buffer of cell #0.
# The output remains in GPU memory in this case.
tx_tensor = srs_tx(srs_tx_config)[0]
tx_tensor = tx_tensor[:, srs_symbols, :]

# Channel transmission using TF and Sionna.
No = pow(10., -esno_db / 10.)
rx_tensor, channel = apply_channel(tx_tensor, No)
rx_tensor = rx_tensor[:, srs_symbols, :]
channel = channel.numpy()

# Run the receiver pipeline.
srs_report = srs_rx([rx_tensor], srs_rx_config)


# ## Plot results
# Plot the sounding results for each PRB group, along with the actual channel realization.

# In[7]:


subc_idx = 6 + np.arange(0, 272 * 12, 2 * 12)
for rx_ant in range(4):
    fig, axs = plt.subplots(2, figsize=(10, 10))
    fig.suptitle(f"SRS channel estimates for Rx antenna {rx_ant}")    
    axs[0].plot(np.real(srs_report[0].ch_est[:, rx_ant, 0]), 'bo', label='MMSE')
    axs[0].plot(np.real(channel[subc_idx, :, rx_ant]), 'k:', label='Channel')
    axs[1].plot(np.imag(srs_report[0].ch_est[:, rx_ant, 0]), 'bo', label='MMSE')
    axs[1].plot(np.imag(channel[subc_idx, :, rx_ant]), 'k:', label='Channel')                   
    axs[0].set_title("Real part")
    axs[1].set_title("Imaginary part")
    for ax in axs:
        ax.grid(True)
        ax.set_xlim(0, 136)
        ax.set_xlabel('PRB group index')
        ax.legend()
    axs[0].grid(True)
    axs[1].grid(True)
    plt.show()


# In[ ]:




