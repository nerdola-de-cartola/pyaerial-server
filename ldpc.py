# ## Imports
import cuda.bindings.runtime as cudart
from collections import defaultdict
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Silence TensorFlow.

import numpy as np
import sionna
import tensorflow as tf

from aerial.phy5g.ldpc import LdpcEncoder
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import LdpcRateMatch
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import CrcEncoder
from aerial.phy5g.ldpc import CrcChecker
from aerial.phy5g.ldpc import get_mcs
from aerial.phy5g.ldpc import random_tb
from simulation_monitor import SimulationMonitor

# Configure the notebook to use only a single GPU and allocate only as much memory as needed.
# For more details, see https://www.tensorflow.org/guide/gpu.
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()               


# ## Parameters
# Set simulation parameters, some numerology parameters, enable/disable scrambling etc.

# Simulation parameters.
#esno_db_range = np.arange(8.1, 8.4, 0.1)
#num_slots = 100
#min_num_tb_errors = 50

# Numerology and frame structure (TS 38.211)
#num_prb = 100              # Number of allocated PRBs (affects TB size and rate-matching)
start_sym = 0              # PDSCH start symbol
num_symbols = 14           # Symbols per slot
num_slots_per_frame = 20   # Slots per frame
#num_layers = #10
dmrs_sym = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

rv = 0

# Rate matching includes scrambling if True
enable_scrambling = True

# Scrambling init based on RNTI and data scrambling ID (TS 38.211)
rnti = 20000
data_scid = 41
cinit = (rnti << 15) + data_scid



# ## Create the LDPC coding chain objects
# The LDPC coding chain objects are created here. This includes the following:
# * `CrcEncoder` which takes the information bits, i.e. the transport block, attaches a transport block CRC into it, segments the TB into code blocks and adds code block CRCs.
# * `LdpcEncoder` which takes the code blocks from `CrcEncoder` as its input, and outputs LDPC encoded code blocks.
# * `LdpcRateMatch` which takes encoded code blocks as its input and outputs a rate matched (and optionally scrambled) stream of bits.
# * `LdpcDerateMatch` which takes the received stream of log-likelihood ratios (LLRs) as its input and outputs derate matched code blocks of LLRs which can be fed to the LDPC decoding. This block performs also descrambling if scrambling is enabled in the pipeline.
# * `LdpcDecoder` which takes the output of LDPC derate matching and decodes the LLRs into code blocks that can then be further concatenated into a received transport block.
# * `CrcChecker` which takes the output of the LDPC decoding block, checks and removes code block CRCs, concatenates code blocks into a full transport block, and finally checks and removed the transport block CRC.
# 
# All components are based on TS 38.212 and thus can be used for transmitting/receiving 5G NR compliant bit streams.
# 
# Also the Sionna channel components and modulation mapper are created here.

# Create also the CUDA stream that running the objects requires.
cudart.cudaSetDevice(0)
cuda_stream = cudart.cudaStreamCreate()[1]
cudart.cudaStreamSynchronize(cuda_stream)

# Create the Aerial Python LDPC objects.
crc_encoder = CrcEncoder(cuda_stream=cuda_stream)
ldpc_encoder = LdpcEncoder(cuda_stream=cuda_stream)
ldpc_decoder = LdpcDecoder(cuda_stream=cuda_stream)
ldpc_rate_match = LdpcRateMatch(enable_scrambling=enable_scrambling, cuda_stream=cuda_stream)
ldpc_derate_match = LdpcDeRateMatch(enable_scrambling=enable_scrambling, cuda_stream=cuda_stream)
crc_checker = CrcChecker(cuda_stream=cuda_stream)


mcs = 20

# Derive modulation order and code rate
mod_order, code_rate = get_mcs(mcs)
print("mod_order", mod_order)
code_rate /= 1024.0

# Create the Sionna modulation mapper/demapper and the AWGN channel.
mapper = sionna.phy.mapping.Mapper("qam", mod_order)
demapper = sionna.phy.mapping.Demapper("app", "qam", mod_order)
channel = sionna.phy.channel.AWGN()


# ## Main simulation loop
def ldpc_stack(esno_db, num_prb, num_layers):                

    # Generate a random transport block (in bits).
    transport_block = random_tb(
        mod_order=mod_order,
        code_rate=code_rate * 1024,
        dmrs_syms=dmrs_sym,
        num_prbs=num_prb,
        start_sym=start_sym,
        num_symbols=num_symbols,
        num_layers=num_layers,
        return_bits=False
    )
    tb_size = transport_block.shape[0] * 8
    
    # Run transport block CRC encoding, code block segmentation and code block CRC encoding.
    code_blocks = crc_encoder.encode(
        tb_inputs=[transport_block],
        tb_sizes=[tb_size],
        code_rates=[code_rate]
    )
    
    # Run the LDPC encoding. The LDPC encoder takes a K x C array as its input, where K is the number of bits per code
    # block and C is the number of code blocks. Its output is N x C where N is the number of coded bits per code block.
    # If there is more than one code block, a code block CRC (random in this case as we do not need an actual CRC) is
    # attached to 
    coded_bits = ldpc_encoder.encode(
        code_blocks=code_blocks,
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        redundancy_versions=[rv]
    )

    # Run rate matching. This needs rate matching length as its input, meaning the number of bits that can be
    # transmitted within the allocated resource elements. The input data is fed as 32-bit floats.        
    num_data_sym = (np.array(dmrs_sym[start_sym:start_sym + num_symbols]) == 0).sum()
    rate_match_len = num_data_sym * num_prb * 12 * num_layers * mod_order
    rate_matched_bits = ldpc_rate_match.rate_match(
        coded_blocks=coded_bits,
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        rate_match_lens=[rate_match_len],
        mod_orders=[mod_order],
        num_layers=[num_layers],
        redundancy_versions=[rv],
        cinits=[cinit]
    )[0]

    # Map the bits to symbols and transmit through an AWGN channel. All this in Sionna.       
    no = sionna.phy.utils.ebnodb2no(esno_db, num_bits_per_symbol=1, coderate=1)
    tx_symbols = mapper(rate_matched_bits[None])        
    rx_symbols = channel(tx_symbols, no)
    llr = -1. * demapper(rx_symbols, no)[0, :].numpy()[:, None]
    
    # Run receiver side (de)rate matching. The input is the received array of bits directly, and the output
    # is a NumPy array of size N x C of log likelihood ratios, represented as 32-bit floats. Descrambling
    # is also performed here in case scrambling is enabled.
    derate_matched_bits = ldpc_derate_match.derate_match(
        input_llrs=[llr],
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        rate_match_lengths=[rate_match_len],
        mod_orders=[mod_order],
        num_layers=[num_layers],
        redundancy_versions=[rv],
        ndis=[1],
        cinits=[cinit]
    )

    # Run LDPC decoding. The decoder takes the derate matching output as its input and returns
    decoded_bits = ldpc_decoder.decode(
        input_llrs=derate_matched_bits,
        tb_sizes=[tb_size],
        code_rates=[code_rate],
        redundancy_versions=[rv],
        rate_match_lengths=[rate_match_len]
    )

    # Combine code blocks into a transport block. CRC ignored as it was just random bits in this example.
    decoded_tb, _ = crc_checker.check_crc(
        input_bits=decoded_bits,
        tb_sizes=[tb_size],
        code_rates=[code_rate]
    )

    num_correct_bits = np.sum(decoded_tb[0] == transport_block)

    num_error_bits = np.sum(decoded_tb[0] != transport_block)

    total_bits = transport_block.size

    ber = num_error_bits / total_bits

    accuracy = num_correct_bits / total_bits

    return ber, accuracy, decoded_bits, derate_matched_bits, rate_matched_bits, coded_bits, code_blocks, transport_block

