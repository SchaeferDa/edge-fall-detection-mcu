#!/bin/bash

stedgeai generate --no-inputs-allocation --no-outputs-allocation --model yolov8n_256_quant_pc_uf_pose_coco-st.tflite --target stm32n6 --st-neural-art default@user_neuralart.json -o .
arm-none-eabi-objcopy -I binary network_atonbuf.xSPI2.raw --change-addresses 0x70380000 -O ihex network_data.hex