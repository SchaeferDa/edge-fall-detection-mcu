#!/bin/bash
# Generate fall detector model for STM32N6 CPU (Cortex-M55, no NPU)
# Output: fall_network.c, fall_network.h, fall_network_data.c, fall_network_data.h

set -e

export PATH=$PATH:/opt/ST/STEdgeAI/3.0/Utilities/linux
export PATH=$PATH:/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

stedgeai generate \
  --no-inputs-allocation \
  --no-outputs-allocation \
  --model fall_model_int8.tflite \
  --target stm32n6 \
  --name fall_network \
  -o .

echo "Done. Generated files:"
ls -la fall_network*.c fall_network*.h 2>/dev/null || true
