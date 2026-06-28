FALL_REL_DIR := $(patsubst %/,%,$(dir $(lastword $(MAKEFILE_LIST))))

C_SOURCES_FALL += $(FALL_REL_DIR)/Src/fall_preprocessor.c
C_SOURCES_FALL += $(FALL_REL_DIR)/Src/fall_classifier.c
C_SOURCES_FALL += $(FALL_REL_DIR)/Model/fall_network.c
C_SOURCES_FALL += $(FALL_REL_DIR)/Model/fall_network_data.c

C_INCLUDES_FALL += -I$(FALL_REL_DIR)/Src
C_INCLUDES_FALL += -I$(FALL_REL_DIR)/Model
C_INCLUDES_FALL += -IInc

C_DEFS_FALL += -DFALL_DETECTION_MODULE

C_SOURCES += $(C_SOURCES_FALL)
C_INCLUDES += $(C_INCLUDES_FALL)
C_DEFS += $(C_DEFS_FALL)
