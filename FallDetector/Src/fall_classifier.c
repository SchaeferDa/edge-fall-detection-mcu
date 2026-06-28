/**
 ******************************************************************************
 * @file    fall_classifier.c
 * @brief   See fall_classifier.h
 ******************************************************************************
 */
#include "fall_classifier.h"

#include <string.h>
#include <math.h>
#include <assert.h>

#include "stai.h"
#include "fall_network.h"

/* ---- input quantisation parameters (from model_int8.tflite) ------------- */
/* serving_default_pose_sequence0: QLinear(0.199651495, 27, int8)            */
#define FALL_IN_SCALE      (0.199651495f)
#define FALL_IN_ZERO_POINT (27)

/* ---- output dequantisation parameters (StatefulPartitionedCall_1:0)       */
/* QLinear(0.003906250, -128, int8)                                           */
#define FALL_OUT_SCALE      (0.003906250f)
#define FALL_OUT_ZERO_POINT (-128)

/* Index 0 = fall, index 1 = neutral (from label_map.json) */
#define FALL_OUT_IDX_FALL    0

/* ---- helpers ------------------------------------------------------------- */

static inline int8_t quantize_f32_to_s8(float v, float scale, int zero_point)
{
  int q = (int)roundf(v / scale) + zero_point;
  if (q < -128) q = -128;
  if (q >  127) q =  127;
  return (int8_t)q;
}

static inline float dequantize_s8_to_f32(int8_t v, float scale, int zero_point)
{
  return scale * (float)((int)v - zero_point);
}

/* ---- public API ---------------------------------------------------------- */

int fall_classifier_init(fall_classifier_t *fc)
{
  stai_return_code ret;
  stai_ptr act_ptr;

  assert(fc->network_ctx);
  assert(fc->activations);
  assert(fc->input_buf);
  assert(fc->output_buf);

  /* stai_runtime_init() is already called by the NN thread before FreeRTOS
   * scheduler starts; do not call it again here.                            */
  ret = stai_fall_network_init((stai_network *)fc->network_ctx);
  if (ret != STAI_SUCCESS) return -1;

  /* Provide user-allocated activations buffer (4800 B).
   * Weights are PREALLOCATED in fall_network_data.c – no set_weights needed. */
  act_ptr = (stai_ptr)fc->activations;
  ret = stai_fall_network_set_activations((stai_network *)fc->network_ctx,
                                          &act_ptr, 1);
  if (ret != STAI_SUCCESS) return -1;

  /* Wire in the persistent input / output buffers */
  {
    stai_ptr inp = (stai_ptr)fc->input_buf;
    ret = stai_fall_network_set_inputs((stai_network *)fc->network_ctx,
                                       &inp, 1);
    if (ret != STAI_SUCCESS) return -1;
  }
  {
    stai_ptr out = (stai_ptr)fc->output_buf;
    ret = stai_fall_network_set_outputs((stai_network *)fc->network_ctx,
                                        &out, 1);
    if (ret != STAI_SUCCESS) return -1;
  }

  fc->consec_fall_count = 0;
  fc->fall_active       = 0;

  return 0;
}

int fall_classifier_run(fall_classifier_t *fc,
                        const float window[FALL_SEQ_LEN][FALL_FEATURES],
                        float *p_fall_out)
{
  stai_return_code ret;
  int8_t  *inp = (int8_t *)fc->input_buf;
  int8_t  *out = (int8_t *)fc->output_buf;
  float    p_fall;
  int      t, f;

  /* 1. Quantise float32 Z-scored window → INT8 input tensor               */
  for (t = 0; t < FALL_SEQ_LEN; t++) {
    for (f = 0; f < FALL_FEATURES; f++) {
      inp[t * FALL_FEATURES + f] =
        quantize_f32_to_s8(window[t][f], FALL_IN_SCALE, FALL_IN_ZERO_POINT);
    }
  }

  /* 2. Run CPU inference (blocking)                                        */
  ret = stai_fall_network_run((stai_network *)fc->network_ctx, STAI_MODE_SYNC);
  if (ret != STAI_SUCCESS) {
    if (p_fall_out) *p_fall_out = 0.0f;
    return 0;
  }

  /* 3. Dequantise output: index 0 = P(fall)                               */
  p_fall = dequantize_s8_to_f32(out[FALL_OUT_IDX_FALL],
                                 FALL_OUT_SCALE, FALL_OUT_ZERO_POINT);
  if (p_fall_out) *p_fall_out = p_fall;

  /* 4. Consecutive-window vote filter                                      */
  if (p_fall >= FALL_DET_THRESHOLD) {
    fc->consec_fall_count++;
    if (fc->consec_fall_count >= FALL_CONSECUTIVE_WIN) {
      fc->fall_active = 1;
      return 1;
    }
  } else {
    fc->consec_fall_count = 0;
    fc->fall_active       = 0;
  }

  return 0;
}

void fall_classifier_reset(fall_classifier_t *fc)
{
  fc->consec_fall_count = 0;
  fc->fall_active       = 0;
}
