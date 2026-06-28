export DKEL="/usr/local/STMicroelectronics/STM32Cube/STM32CubeProgrammer/bin/ExternalLoader/MX66UW1G45G_STM32N6570-DK.stldr"

# STM32_SigningTool_CLI -bin build/Project.bin -nk -s -t ssbl -hv 2.3 -o build/Project_sign.bin
STM32_SigningTool_CLI -align -bin build/Project.bin -nk -s -t ssbl -hv 2.3 -o build/Project_sign.bin

# First Stage Boot Loader
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el $DKEL -hardRst -w FSBL/ai_fsbl.hex

# Network Parameters and Biases
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el $DKEL -hardRst -w Model/network_data.hex

# Adapt build path to your IDE
STM32_Programmer_CLI -c port=SWD mode=HOTPLUG -el $DKEL -hardRst -w build/Project_sign.bin 0x70100000