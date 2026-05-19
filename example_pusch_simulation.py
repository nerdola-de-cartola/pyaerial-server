#!/usr/bin/env python
# coding: utf-8

# # Using pyAerial to run a PUSCH link simulation
# This example shows how to use the pyAerial cuPHY Python bindings to run a PUSCH link simulation. PUSCH transmitter is emulated by PDSCH transmission with properly chosen parameters, that way making it a 5G NR compliant PUSCH transmission. Building a PUSCH receiver using pyAerial is demonstrated in two ways, first by using a fully fused, complete, PUSCH receiver called from Python using just a single function call. The same is then achieved by building the complete PUSCH receiver using individual separate Python function calls to individual PUSCH receiver components.
# 
# The NVIDIA [Sionna](https://nvlabs.github.io/sionna/) library is utilized for simulating the radio channel based on 3GPP channel models.

# ## Imports

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'widget')
import datetime
from collections import defaultdict
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Silence TensorFlow.

import numpy as np
import cupy as cp
import sionna
import tensorflow as tf

from aerial.phy5g.pdsch import PdschTx
from aerial.phy5g.pdsch import PdschTxPipelineFactory
from aerial.phy5g.pusch import PuschRx
from aerial.phy5g.pusch import PuschRxPipelineFactory
from aerial.phy5g.pusch import SeparablePuschRx
from aerial.phy5g.pusch import SeparablePuschRxPipelineFactory
from aerial.phy5g.config import AerialPuschRxConfig
from aerial.phy5g.config import AerialPdschTxConfig
from aerial.phy5g.config import PdschConfig
from aerial.phy5g.config import PdschUeConfig
from aerial.phy5g.config import PdschCwConfig
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.phy5g.ldpc import get_mcs
from aerial.phy5g.ldpc import get_tb_size
from aerial.phy5g.ldpc import random_tb
from aerial.util.cuda import get_cuda_stream
from aerial.pycuphy.types import PuschLdpcKernelLaunch
from simulation_monitor import SimulationMonitor


# Configure the notebook to use only a single GPU and allocate only as much memory as needed.
# For more details, see https://www.tensorflow.org/guide/gpu.
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


# ## Parameters
# Set simulation parameters, numerology, PUSCH parameters and channel parameters here.

# In[2]:


# Simulation parameters.
use_cupy = True  # Use NumPy or CuPy - with NumPy there are H2D/D2H copies between every PUSCH receiver component, resulting in slower simulation.
esno_db_range = np.arange(-5.6, -4.0, 0.2)
num_slots = 10000
min_num_tb_errors = 250

# Numerology and frame structure. See TS 38.211.
num_ofdm_symbols = 14
fft_size = 4096
cyclic_prefix_length = 288
subcarrier_spacing = 30e3
num_guard_subcarriers = (410, 410)
num_slots_per_frame = 20

# System/gNB configuration
num_tx_ant = 1             # UE antennas
num_rx_ant = 2             # gNB antennas
cell_id = 41               # Physical cell ID
enable_pusch_tdi = 0       # Enable time interpolation for equalizer coefficients
eq_coeff_algo = 1          # Equalizer algorithm

# PUSCH configuration
rnti = 1234                # UE RNTI
scid = 0                   # DMRS scrambling ID
data_scid = 0              # Data scrambling ID
layers = 1                 # Number of layers
mcs_index = 2              # MCS index as per TS 38.214 table. Note: Es/No range may need to be changed too to get meaningful results.
mcs_table = 0              # MCS table index
dmrs_ports = 1             # Used DMRS port.
start_prb = 0              # Start PRB index.
num_prbs = 273             # Number of allocated PRBs.
start_sym = 2              # Start symbol index.
num_symbols = 12           # Number of symbols.
dmrs_scrm_id = 41          # DMRS scrambling ID
dmrs_syms = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Indicates which symbols are used for DMRS.
dmrs_max_len = 1
dmrs_add_ln_pos = 0
num_dmrs_cdm_grps_no_data = 2
precoding_matrix = None
mod_order, code_rate = get_mcs(mcs_index, mcs_table+1)  # Different indexing for MCS table.
tb_size = get_tb_size(  # TB size in bits
    mod_order=mod_order,
    code_rate=code_rate,
    dmrs_syms=dmrs_syms,
    num_prbs=num_prbs,
    start_sym=start_sym,
    num_symbols=num_symbols,
    num_layers=layers)

# Channel parameters
carrier_frequency = 3.5e9  # Carrier frequency in Hz.
delay_spread = 100e-9      # Nominal delay spread in [s]. Please see the CDL documentation
                           # about how to choose this value.
link_direction = "uplink"
channel_model = "Rayleigh" # Channel model: Suitable values:
                           # "Rayleigh" - Rayleigh block fading channel model (sionna.channel.RayleighBlockFading)
                           # "CDL-x", where x is one of ["A", "B", "C", "D", "E"] - for 3GPP CDL channel models
                           #          as per TR 38.901.
speed = 0.8333             # UE speed [m/s]. The direction of travel will chosen randomly within the x-y plane.


# ## Create the pipelines
# As mentioned, PUSCH transmission is emulated here by the PDSCH transmission chain. Note that the static cell parameters and static PUSCH parameters are given upon creating the PUSCH transmission/reception objects. Dynamically (per slot) changing parameters are however set when actually running the transmission/reception, see further below.

# In[3]:


cuda_stream = get_cuda_stream()

# PDSCH configuration objects. PDSCH transmitter emulates PUSCH transmission here.
# Note that default values are used for some parameters not given here.
pdsch_cw_config = PdschCwConfig(
    mcs_table=mcs_table,
    mcs_index=mcs_index,
    code_rate=int(code_rate * 10),
    mod_order=mod_order
)
pdsch_ue_config = PdschUeConfig(
    cw_configs=[pdsch_cw_config],
    scid=scid,
    dmrs_scrm_id=dmrs_scrm_id,
    layers=layers,
    dmrs_ports=dmrs_ports,
    rnti=rnti,
    data_scid=data_scid,
    precoding_matrix=precoding_matrix
)
pdsch_config = PdschConfig(
    ue_configs=[pdsch_ue_config],
    num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,

    start_prb=start_prb,
    num_prbs=num_prbs,
    dmrs_syms=dmrs_syms,
    start_sym=start_sym,
    num_symbols=num_symbols    
)

pdsch_tx_config = AerialPdschTxConfig(
    cell_id=cell_id,
    num_tx_ant=num_tx_ant
)

# Transmitter pipeline.
tx_pipeline = PdschTxPipelineFactory().create(pdsch_tx_config, cuda_stream)

# PUSCH configuration objects. Note that default values are used for some parameters
# not given here.
pusch_ue_config = PuschUeConfig(
    scid=scid,
    layers=layers,
    dmrs_ports=dmrs_ports,
    rnti=rnti,
    data_scid=data_scid,
    mcs_table=mcs_table,
    mcs_index=mcs_index,
    code_rate=int(code_rate * 10),
    mod_order=mod_order,
    tb_size=tb_size // 8
)
# Note that this is a list. One UE group only in this case.
pusch_configs = [PuschConfig(
    ue_configs=[pusch_ue_config],
    num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
    dmrs_scrm_id=dmrs_scrm_id,
    start_prb=start_prb,
    num_prbs=num_prbs,
    dmrs_syms=dmrs_syms,
    dmrs_max_len=dmrs_max_len,
    dmrs_add_ln_pos=dmrs_add_ln_pos,
    start_sym=start_sym,
    num_symbols=num_symbols
)]

pusch_rx_config = AerialPuschRxConfig(
    cell_id=cell_id,
    num_rx_ant=num_rx_ant,
    enable_pusch_tdi=enable_pusch_tdi,
    eq_coeff_algo=eq_coeff_algo,
    ldpc_kernel_launch=PuschLdpcKernelLaunch.PUSCH_RX_LDPC_STREAM_SEQUENTIAL
)

cases = {
    "Fused": PuschRxPipelineFactory,
    "Separable": SeparablePuschRxPipelineFactory
}

pipelines = {}
for name, factory in cases.items():
    pipelines[name] = factory().create(pusch_rx_config, cuda_stream)


# ## Channel generation using Sionna
# Simulating the transmission through the radio channel takes advantage of the channel model implementations available in NVIDIA Sionna. In Sionna, the transmission can be simulated directly in frequency domain by defining a resource grid. In our case, reference signal patterns and data carrying resource elements are defined elsewhere within the Aerial code, hence we define resource grid as a simple dummy grid containing only data symbols.
# 
# See also: [Sionna documentation](https://nvlabs.github.io/sionna/index.html)

# In[4]:


# Define the resource grid.
resource_grid = sionna.phy.ofdm.ResourceGrid(
    num_ofdm_symbols=num_ofdm_symbols,
    fft_size=fft_size,
    subcarrier_spacing=subcarrier_spacing,
    num_tx=num_tx_ant,
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
    num_cols=int(num_rx_ant/2),
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

channel = sionna.phy.channel.OFDMChannel(
    ch_model,
    resource_grid,
    add_awgn=True,
    normalize_channel=True,
    return_channel=False
)

def apply_channel(tx_tensor, No):
    """Transmit the Tx tensor through the radio channel."""
    # Add batch and num_streams dimensions that Sionna expects and reshape.
    tx_tensor = tf.transpose(tx_tensor, (2, 1, 0))
    tx_tensor = tf.reshape(tx_tensor, (num_tx_ant, -1))[None, :, None, :]
    tx_tensor = resource_grid_mapper(tx_tensor)
    rx_tensor = channel(tx_tensor, No)
    rx_tensor = remove_guard_subcarriers(rx_tensor)
    rx_tensor = rx_tensor[0, 0]
    rx_tensor = tf.transpose(rx_tensor, (2, 1, 0))
    return rx_tensor


# ## Run the actual simulation
# Here we loop across the Es/No range, and simulate a number of slots for each Es/No value. A single transport block is simulated within a slot. The simulation starts over from the next Es/No value when a minimum number of transport block errors is reached.

# In[5]:


monitor = SimulationMonitor(cases, esno_db_range)
exec_times = dict.fromkeys(cases, 0)

# Loop the Es/No range.
bler = []
for esno_db in esno_db_range:
    monitor.step(esno_db)
    num_tb_errors = defaultdict(int)

    # Run multiple slots and compute BLER.
    for slot_idx in range(num_slots):
        slot_number = slot_idx % num_slots_per_frame

        # Get random transport block.
        tb_input_np = random_tb(
            mod_order=mod_order,
            code_rate=code_rate,
            dmrs_syms=dmrs_syms,
            num_prbs=num_prbs,
            start_sym=start_sym,
            num_symbols=num_symbols,
            num_layers=layers)

        if use_cupy:
            tb_input = cp.array(tb_input_np, dtype=cp.uint8, order='F')
        else:
            tb_input = tb_input_np
                           
        # Transmit PUSCH. This is where we set the dynamically changing parameters.
        # Input parameters are given as lists as the interface supports multiple UEs.
        tx_tensor = tx_pipeline(
            slot=slot_number,
            tb_inputs=[tb_input],
            config=[pdsch_config]
        )
       
        # Channel transmission using TF and Sionna. Note: Some conversions are necessary if we are
        # using NumPy-based API.
        No = pow(10., -esno_db / 10.)
        if use_cupy:
            tx_tensor = tf.experimental.dlpack.from_dlpack(tx_tensor.toDlpack())
        rx_tensor = apply_channel(tx_tensor, No)
        
        if use_cupy:
            rx_tensor = tf.experimental.dlpack.to_dlpack(rx_tensor)
            rx_tensor = cp.from_dlpack(rx_tensor)
        else:
            rx_tensor = np.array(rx_tensor)

        # Run the PUSCH receiver pipelines.
        # Note that this is where we set the dynamically changing parameters.
        for name, pipeline in pipelines.items():
            start_time = time.time()
            tb_crcs, tbs = pipeline(
                slot=slot_number,
                rx_slot=rx_tensor,
                config=pusch_configs
            )
            exec_times[name] += time.time() - start_time            
            num_tb_errors[name] += int(np.array_equal(tbs[0], tb_input_np) == False)

        monitor.update(num_tbs=slot_idx + 1, num_tb_errors=num_tb_errors)
        if (np.array(list(num_tb_errors.values())) >= min_num_tb_errors).all():
            break  # Next Es/No value.

    monitor.finish_step(num_tbs=slot_idx + 1, num_tb_errors=num_tb_errors)
monitor.finish()


# In[ ]:


exec_times = {k : v / (num_slots * len(esno_db_range)) for k, v in exec_times.items()}
print("Average execution times:")
for name, exec_time in exec_times.items():
    print(f"{name}: {exec_time * 1000: .2f} ms.")


# In[ ]:




