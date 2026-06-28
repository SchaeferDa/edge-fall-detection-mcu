/**
 ******************************************************************************
 * @file    fall_preprocessor.h
 * @brief   Preprocessing pipeline for fall detection:
 *          hip-centering, shoulder-scale normalization, sliding window,
 *          Z-score normalization.
 *
 *  Pipeline order (must be exact):
 *    1. Scale keypoints to 256x256 space
 *    2. Hip-centering
 *    3. Shoulder-scale (rolling median over last FALL_SHOULDER_HIST frames)
 *    4. Zero-out low-confidence keypoints
 *    5. Flatten into sliding window (FIFO, 30 frames x 51 features)
 *    6. Z-score normalisation using pre-computed mean/std (fall_scaler_data.h)
 ******************************************************************************
 */
#ifndef FALL_PREPROCESSOR_H
#define FALL_PREPROCESSOR_H

#include <stdint.h>
#include "mpe_pp_output_if.h"

/* ---- tuneable constants -------------------------------------------------- */
#define FALL_KP_COUNT          17   /* COCO keypoints                          */
#define FALL_FEATURES          51   /* 17 * 3 (x, y, conf)                    */
#define FALL_SEQ_LEN           30   /* sliding window length (frames)          */
#define FALL_STRIDE             5   /* run classifier every N frames           */
#define FALL_CONF_THRESH       0.3f /* keypoint confidence threshold           */
#define FALL_MIN_SHOULDER_DIST 3.0f /* minimum valid shoulder distance (px)   */
#define FALL_SHOULDER_HIST     90   /* rolling-median history depth            */
#define FALL_INPUT_SIZE_PX   256.0f /* normalised space [0,256]               */

/* ---- public state -------------------------------------------------------- */
typedef struct {
  /* rolling-median shoulder distance buffer */
  float  shoulder_hist[FALL_SHOULDER_HIST];
  int    shoulder_hist_cnt;   /* number of valid entries so far */
  int    shoulder_hist_head;  /* next write position (ring)     */

  /* previous-frame keypoints for temporal interpolation */
  float  prev_kp[FALL_KP_COUNT][3]; /* [i][0]=x, [i][1]=y, [i][2]=conf */
  int    prev_kp_valid;

  /* sliding-window FIFO, shape [FALL_SEQ_LEN][FALL_FEATURES] */
  float  window[FALL_SEQ_LEN][FALL_FEATURES];
  int    window_fill;         /* frames written (clamped to FALL_SEQ_LEN)     */

  /* frame counter for stride logic */
  int    frame_counter;
} fall_preprocessor_t;

/* ---- API ----------------------------------------------------------------- */

/**
 * @brief  Reset all preprocessor state.
 */
void fall_preprocessor_init(fall_preprocessor_t *pp);

/**
 * @brief  Process one new frame from the YOLO post-processor.
 *
 *  The function scales, centres, and normalises the dominant person's
 *  keypoints, pushes them into the sliding window and, when enough frames
 *  and the stride condition are met, copies the Z-scored window into
 *  @p window_out.
 *
 * @param  pp          Preprocessor state.
 * @param  detects     Detection array from mpe post-processing.
 * @param  nb_detect   Number of detections.
 * @param  window_out  Output buffer, shape [FALL_SEQ_LEN][FALL_FEATURES].
 *                     Only valid when the function returns 1.
 * @return 1  Window is ready for inference (stride condition met, buffer full).
 *         0  Not enough frames yet or stride condition not met.
 */
int fall_preprocessor_push(fall_preprocessor_t *pp,
                           const mpe_pp_outBuffer_t *detects,
                           int nb_detect,
                           float window_out[FALL_SEQ_LEN][FALL_FEATURES]);

#endif /* FALL_PREPROCESSOR_H */
