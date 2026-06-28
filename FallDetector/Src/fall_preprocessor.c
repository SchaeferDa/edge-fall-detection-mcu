/**
 ******************************************************************************
 * @file    fall_preprocessor.c
 * @brief   See fall_preprocessor.h
 ******************************************************************************
 */
#include "fall_preprocessor.h"
#include "fall_scaler_data.h"

#include <string.h>
#include <math.h>
#include <stddef.h>

/* ---- index constants (COCO keypoint layout) ------------------------------ */
#define IDX_NOSE           0
#define IDX_LEFT_SHOULDER  5
#define IDX_RIGHT_SHOULDER 6
#define IDX_LEFT_HIP      11
#define IDX_RIGHT_HIP     12

/* ---- static helpers ------------------------------------------------------ */

static float s_fabsf(float v) { return v < 0.0f ? -v : v; }

/* Insertion-sort based median for small arrays (N <= FALL_SHOULDER_HIST).    */
static float median_of(const float *arr, int n)
{
  /* copy + insertion sort */
  float tmp[FALL_SHOULDER_HIST];
  int i, j;
  for (i = 0; i < n; i++) tmp[i] = arr[i];
  for (i = 1; i < n; i++) {
    float key = tmp[i];
    for (j = i - 1; j >= 0 && tmp[j] > key; j--)
      tmp[j + 1] = tmp[j];
    tmp[j + 1] = key;
  }
  if (n % 2 == 1)
    return tmp[n / 2];
  return (tmp[n / 2 - 1] + tmp[n / 2]) * 0.5f;
}

/* ---- public API ---------------------------------------------------------- */

void fall_preprocessor_init(fall_preprocessor_t *pp)
{
  memset(pp, 0, sizeof(*pp));
}

int fall_preprocessor_push(fall_preprocessor_t *pp,
                           const mpe_pp_outBuffer_t *detects,
                           int nb_detect,
                           float window_out[FALL_SEQ_LEN][FALL_FEATURES])
{
  int i, t, f;

  /* ------------------------------------------------------------------ */
  /* 1. Find dominant person (highest mean keypoint confidence)          */
  /* ------------------------------------------------------------------ */
  int dom = -1;
  float dom_conf = -1.0f;
  for (i = 0; i < nb_detect; i++) {
    float sum = 0.0f;
    int j;
    for (j = 0; j < FALL_KP_COUNT; j++)
      sum += detects[i].pKeyPoints[j].conf;
    float avg = sum / (float)FALL_KP_COUNT;
    if (avg > dom_conf) {
      dom_conf = avg;
      dom = i;
    }
  }

  /* ------------------------------------------------------------------ */
  /* 2. Build raw keypoint array in 256x256 space                        */
  /*    Use temporal interpolation for missing keypoints.                */
  /* ------------------------------------------------------------------ */
  float kp[FALL_KP_COUNT][3]; /* [i][0]=x  [i][1]=y  [i][2]=conf */

  if (dom >= 0) {
    const mpe_pp_outBuffer_t *d = &detects[dom];
    for (i = 0; i < FALL_KP_COUNT; i++) {
      /* Keypoints from post-processor are in [0,1] – scale to [0,256] */
      kp[i][0] = d->pKeyPoints[i].x    * FALL_INPUT_SIZE_PX;
      kp[i][1] = d->pKeyPoints[i].y    * FALL_INPUT_SIZE_PX;
      kp[i][2] = d->pKeyPoints[i].conf;

      /* Temporal interpolation: reuse previous frame for low-conf kp   */
      if (kp[i][2] < FALL_CONF_THRESH && pp->prev_kp_valid) {
        kp[i][0] = pp->prev_kp[i][0];
        kp[i][1] = pp->prev_kp[i][1];
        kp[i][2] = 0.0f; /* mark still-interpolated */
      }
    }
  } else {
    /* No detection: use previous frame or zeros */
    if (pp->prev_kp_valid) {
      for (i = 0; i < FALL_KP_COUNT; i++) {
        kp[i][0] = pp->prev_kp[i][0];
        kp[i][1] = pp->prev_kp[i][1];
        kp[i][2] = 0.0f;
      }
    } else {
      memset(kp, 0, sizeof(kp));
    }
  }

  /* ------------------------------------------------------------------ */
  /* 3. Hip-centering                                                    */
  /* ------------------------------------------------------------------ */
  float hip_x = 0.0f, hip_y = 0.0f;
  int lh_ok = (kp[IDX_LEFT_HIP][2]  > FALL_CONF_THRESH) ? 1 : 0;
  int rh_ok = (kp[IDX_RIGHT_HIP][2] > FALL_CONF_THRESH) ? 1 : 0;

  if (lh_ok && rh_ok) {
    hip_x = (kp[IDX_LEFT_HIP][0] + kp[IDX_RIGHT_HIP][0]) * 0.5f;
    hip_y = (kp[IDX_LEFT_HIP][1] + kp[IDX_RIGHT_HIP][1]) * 0.5f;
  } else if (lh_ok) {
    hip_x = kp[IDX_LEFT_HIP][0];
    hip_y = kp[IDX_LEFT_HIP][1];
  } else if (rh_ok) {
    hip_x = kp[IDX_RIGHT_HIP][0];
    hip_y = kp[IDX_RIGHT_HIP][1];
  }
  /* else: no valid hip → hip_x/y stay 0 */

  for (i = 0; i < FALL_KP_COUNT; i++) {
    kp[i][0] -= hip_x;
    kp[i][1] -= hip_y;
  }

  /* ------------------------------------------------------------------ */
  /* 4. Shoulder-scale: update rolling-median, then normalise           */
  /* ------------------------------------------------------------------ */
  float ls_conf = kp[IDX_LEFT_SHOULDER][2];
  float rs_conf = kp[IDX_RIGHT_SHOULDER][2];
  if (ls_conf > FALL_CONF_THRESH && rs_conf > FALL_CONF_THRESH) {
    float dx = kp[IDX_LEFT_SHOULDER][0] - kp[IDX_RIGHT_SHOULDER][0];
    float dy = kp[IDX_LEFT_SHOULDER][1] - kp[IDX_RIGHT_SHOULDER][1];
    float dist = sqrtf(dx * dx + dy * dy);
    if (dist > FALL_MIN_SHOULDER_DIST) {
      /* add to ring buffer */
      pp->shoulder_hist[pp->shoulder_hist_head] = dist;
      pp->shoulder_hist_head = (pp->shoulder_hist_head + 1) % FALL_SHOULDER_HIST;
      if (pp->shoulder_hist_cnt < FALL_SHOULDER_HIST)
        pp->shoulder_hist_cnt++;
    }
  }

  if (pp->shoulder_hist_cnt > 0) {
    float scale = median_of(pp->shoulder_hist, pp->shoulder_hist_cnt);
    if (scale > 1e-6f) {
      for (i = 0; i < FALL_KP_COUNT; i++) {
        kp[i][0] /= scale;
        kp[i][1] /= scale;
        /* conf unchanged */
      }
    }
  }

  /* ------------------------------------------------------------------ */
  /* 5. Zero-out low-confidence keypoints (AFTER centering + scaling)   */
  /* ------------------------------------------------------------------ */
  for (i = 0; i < FALL_KP_COUNT; i++) {
    if (kp[i][2] < FALL_CONF_THRESH) {
      kp[i][0] = 0.0f;
      kp[i][1] = 0.0f;
    }
  }

  /* ------------------------------------------------------------------ */
  /* 6. Save to previous-frame buffer                                    */
  /* ------------------------------------------------------------------ */
  for (i = 0; i < FALL_KP_COUNT; i++) {
    pp->prev_kp[i][0] = kp[i][0];
    pp->prev_kp[i][1] = kp[i][1];
    pp->prev_kp[i][2] = kp[i][2];
  }
  pp->prev_kp_valid = 1;

  /* ------------------------------------------------------------------ */
  /* 7. Flatten into sliding window (FIFO shift)                         */
  /* ------------------------------------------------------------------ */
  /* Shift existing frames down by one */
  if (pp->window_fill >= FALL_SEQ_LEN) {
    memmove(&pp->window[0], &pp->window[1],
            (FALL_SEQ_LEN - 1) * FALL_FEATURES * sizeof(float));
  }

  /* Write new frame at the last position */
  {
    int slot = (pp->window_fill < FALL_SEQ_LEN)
               ? pp->window_fill
               : FALL_SEQ_LEN - 1;
    for (i = 0; i < FALL_KP_COUNT; i++) {
      pp->window[slot][i * 3 + 0] = kp[i][0];
      pp->window[slot][i * 3 + 1] = kp[i][1];
      pp->window[slot][i * 3 + 2] = kp[i][2];
    }
    if (pp->window_fill < FALL_SEQ_LEN)
      pp->window_fill++;
  }

  /* ------------------------------------------------------------------ */
  /* 8. Stride gating: run inference every FALL_STRIDE frames            */
  /* ------------------------------------------------------------------ */
  pp->frame_counter++;
  if (pp->window_fill < FALL_SEQ_LEN ||
      (pp->frame_counter % FALL_STRIDE) != 0) {
    return 0;
  }

  /* ------------------------------------------------------------------ */
  /* 9. Z-score normalisation into output buffer                         */
  /* ------------------------------------------------------------------ */
  for (t = 0; t < FALL_SEQ_LEN; t++) {
    for (f = 0; f < FALL_FEATURES; f++) {
      window_out[t][f] =
        (pp->window[t][f] - fall_scaler_mean[t][f]) / fall_scaler_std[t][f];
    }
  }

  return 1;
}
