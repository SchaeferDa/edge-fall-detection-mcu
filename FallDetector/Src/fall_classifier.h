/**
 ******************************************************************************
 * @file    fall_classifier.h
 * @brief   Fall detection classifier wrapper around the STEdgeAI-generated
 *          fall_network (CPU, Cortex-M55, INT8).
 *
 *  Input:  Z-scored float32 window, shape [FALL_SEQ_LEN][FALL_FEATURES]
 *  Output: boolean fall detection result + raw probability P(fall)
 *
 *  Uses a consecutive-window vote filter to reduce false positives.
 ******************************************************************************
 */
#ifndef FALL_CLASSIFIER_H
#define FALL_CLASSIFIER_H

#include <stdint.h>
#include "fall_preprocessor.h"  /* for FALL_SEQ_LEN, FALL_FEATURES */

/* ---- tuneable constants -------------------------------------------------- */
#ifndef FALL_DET_THRESHOLD
#define FALL_DET_THRESHOLD     0.5f   /* P(fall) decision threshold           */
#endif

#ifndef FALL_CONSECUTIVE_WIN
#define FALL_CONSECUTIVE_WIN   3      /* consecutive positive windows required */
#endif

/* ---- public state -------------------------------------------------------- */
typedef struct {
  /* opaque STEdgeAI network context (must be STAI_FALL_NETWORK_CONTEXT_SIZE) */
  uint8_t  *network_ctx;
  /* activations buffer (must be STAI_FALL_NETWORK_ACTIVATIONS_SIZE bytes)   */
  uint8_t  *activations;
  /* INT8 input  buffer (must be STAI_FALL_NETWORK_IN_SIZE_BYTES bytes)      */
  uint8_t  *input_buf;
  /* INT8 output buffer (must be STAI_FALL_NETWORK_OUT_SIZE_BYTES bytes)     */
  uint8_t  *output_buf;

  /* consecutive positive window counter */
  int      consec_fall_count;
  /* current fall detection state (latched until reset) */
  int      fall_active;
} fall_classifier_t;

/* ---- API ----------------------------------------------------------------- */

/**
 * @brief  Initialise fall classifier. Must be called once before use.
 *
 *  The caller must set all buffer pointers in @p fc before calling this.
 *  Required sizes are available as macros in fall_network.h:
 *    STAI_FALL_NETWORK_CONTEXT_SIZE
 *    STAI_FALL_NETWORK_ACTIVATIONS_SIZE_BYTES
 *    STAI_FALL_NETWORK_IN_SIZE_BYTES
 *    STAI_FALL_NETWORK_OUT_SIZE_BYTES
 *
 * @param  fc  Classifier state with buffer pointers already set.
 * @return 0 on success, -1 on error.
 */
int fall_classifier_init(fall_classifier_t *fc);

/**
 * @brief  Run inference on a Z-scored window.
 *
 * @param  fc          Classifier state.
 * @param  window      Z-scored input, shape [FALL_SEQ_LEN][FALL_FEATURES].
 * @param  p_fall_out  Optional: receives raw P(fall) probability [0..1].
 * @return 1  Fall event confirmed (FALL_CONSECUTIVE_WIN windows triggered).
 *         0  No fall event.
 */
int fall_classifier_run(fall_classifier_t *fc,
                        const float window[FALL_SEQ_LEN][FALL_FEATURES],
                        float *p_fall_out);

/**
 * @brief  Reset the consecutive-window counter and fall_active state.
 */
void fall_classifier_reset(fall_classifier_t *fc);

#endif /* FALL_CLASSIFIER_H */
