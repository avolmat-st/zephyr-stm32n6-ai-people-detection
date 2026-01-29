/**
 ******************************************************************************
 * @file    model.c
 * @author  GPM Application Team
 *
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */

#include "model.h"

#include <stdlib.h>

#include <zephyr/cache.h>
#include <zephyr/drivers/video.h>
#include <zephyr/logging/log.h>
#include <zephyr/mem_mgmt/mem_attr_heap.h>

#include "npu_cache.h"
#include "stai.h"
#include "stai_network.h"
#include "od_st_yolox_pp_if.h"

#define DISPLAY_WIDTH DT_PROP(DT_CHOSEN(zephyr_display), width)
#define DISPLAY_HEIGHT DT_PROP(DT_CHOSEN(zephyr_display), height)

/* post process conf */
 /* Tunable post process parameters */
#define AI_OD_ST_YOLOX_PP_IOU_THRESHOLD             (0.5)
#define AI_OD_ST_YOLOX_PP_CONF_THRESHOLD            (0.6)
#define AI_OD_ST_YOLOX_PP_MAX_BOXES_LIMIT           (10)
 /* Model related defines */
#define AI_OD_ST_YOLOX_PP_NB_CLASSES                (1)
#define AI_OD_ST_YOLOX_PP_L_GRID_WIDTH              (52)
#define AI_OD_ST_YOLOX_PP_L_GRID_HEIGHT             (52)
#define AI_OD_ST_YOLOX_PP_M_GRID_WIDTH              (26)
#define AI_OD_ST_YOLOX_PP_M_GRID_HEIGHT             (26)
#define AI_OD_ST_YOLOX_PP_S_GRID_WIDTH              (13)
#define AI_OD_ST_YOLOX_PP_S_GRID_HEIGHT             (13)
#define AI_OD_ST_YOLOX_PP_NB_ANCHORS                (3)
/* Anchor boxes */
static const float32_t AI_OD_ST_YOLOX_PP_L_ANCHORS[2 * AI_OD_ST_YOLOX_PP_NB_ANCHORS] = {26.000000, 26.000000, 3.640000, 13.000000, 11.960000, 36.40000};
static const float32_t AI_OD_ST_YOLOX_PP_M_ANCHORS[2 * AI_OD_ST_YOLOX_PP_NB_ANCHORS] = {13.000000, 13.000000, 1.820000,  6.500000,  5.980000, 18.20000};
static const float32_t AI_OD_ST_YOLOX_PP_S_ANCHORS[2 * AI_OD_ST_YOLOX_PP_NB_ANCHORS] = { 6.500000,  6.500000, 0.910000,  3.250000,  2.990000,  9.10000};


LOG_MODULE_REGISTER(model);

/* Model instance buffer */
static uint8_t network_ctx[STAI_NETWORK_CONTEXT_SIZE] __attribute__ ((aligned (32)));

/* Post process */
static stai_network_info *current_info;
static size_t output_order_index[3];

/* latest detections and associate mutex */
static od_pp_outBuffer_t pp_detections_buffer[AI_OD_ST_YOLOX_PP_NB_ANCHORS *
					      MAX(AI_OD_ST_YOLOX_PP_MAX_BOXES_LIMIT,
						  AI_OD_ST_YOLOX_PP_L_GRID_WIDTH * AI_OD_ST_YOLOX_PP_L_GRID_HEIGHT +
						  AI_OD_ST_YOLOX_PP_M_GRID_WIDTH * AI_OD_ST_YOLOX_PP_M_GRID_HEIGHT +
						  AI_OD_ST_YOLOX_PP_S_GRID_WIDTH * AI_OD_ST_YOLOX_PP_S_GRID_HEIGHT)];
static od_pp_outBuffer_t detections[AI_OD_ST_YOLOX_PP_MAX_BOXES_LIMIT];
static int detections_nb = -1;
static struct k_mutex boxes_lock;

static int latest_inference_time = 0;

static void Run_Inference(stai_network *network_instance)
{
	stai_return_code ret;

	do {
		ret = stai_network_run(network_instance, STAI_MODE_ASYNC);
		if (ret == STAI_RUNNING_WFE)
			LL_ATON_OSAL_WFE();
	} while (ret == STAI_RUNNING_WFE || ret == STAI_RUNNING_NO_WFE);

	ret = stai_ext_network_new_inference(network_instance);
	assert(ret == STAI_SUCCESS);
}


static void NPUCache_config()
{
	npu_cache_enable();
}

static void model_npu_init()
{
	NPUCache_config();

	LOG_INF("Npu subsystem ready");
}

static int clamp_point(int *x, int *y)
{
	int xi = *x;
	int yi = *y;

	if (*x < 0)
		*x = 0;
	if (*y < 0)
		*y = 0;
	if (*x >= DISPLAY_WIDTH)
		*x = DISPLAY_WIDTH - 1;
	if (*y >= DISPLAY_HEIGHT)
		*y = DISPLAY_HEIGHT - 1;

	return (xi != *x) || (yi != *y);
}

static void convert_length(float32_t wi, float32_t hi, int *wo, int *ho)
{
	*wo = (int) (DISPLAY_WIDTH * wi);
	*ho = (int) (DISPLAY_HEIGHT * hi);
}

static void convert_point(float32_t xi, float32_t yi, int *xo, int *yo)
{
	*xo = (int) (DISPLAY_WIDTH * xi);
	*yo = (int) (DISPLAY_HEIGHT * yi);
}

static void model_detection_to_box(od_pp_outBuffer_t *d, struct dbox *b)
{
	int xc, yc;
	int x0, y0;
	int x1, y1;
	int w, h;

	convert_point(d->x_center, d->y_center, &xc, &yc);
	convert_length(d->width, d->height, &w, &h);
	x0 = xc - (w + 1) / 2;
	y0 = yc - (h + 1) / 2;
	x1 = xc + (w + 1) / 2;
	y1 = yc + (h + 1) / 2;
	clamp_point(&x0, &y0);
	clamp_point(&x1, &y1);
	w = x1 - x0 + 1;
	h = y1 - y0 + 1;

	b->x = x0;
	b->y = y0;
	b->w = w;
	b->h = h;
}

static int model_pp_compare_order_by_ascending_order(const void *el1, const void *el2)
{
  stai_network_info *info = current_info;
  size_t *idx1 = (size_t *) el1;
  size_t *idx2 = (size_t *) el2;

  return info->outputs[*idx1].size_bytes - info->outputs[*idx2].size_bytes;
}

static void model_pp_order_outputs(size_t *outputs_idx, size_t size, stai_network_info *info)
{
  for (size_t i = 0; i < size; i++) {
    outputs_idx[i] = i;
  }

  current_info = info;
  qsort(outputs_idx, size, sizeof(size_t), model_pp_compare_order_by_ascending_order);
  current_info = NULL;
}

static void model_pp_init(od_st_yolox_pp_static_param_t *params, stai_network_info *info)
{
	int ret;

	/* Model output orders may change. So order them by size */
	model_pp_order_outputs(output_order_index, 3, info);

	params->nb_classes = AI_OD_ST_YOLOX_PP_NB_CLASSES;
	params->nb_anchors = AI_OD_ST_YOLOX_PP_NB_ANCHORS;
	params->grid_width_S = AI_OD_ST_YOLOX_PP_S_GRID_WIDTH;
	params->grid_height_S = AI_OD_ST_YOLOX_PP_S_GRID_HEIGHT;
	params->pAnchors_S = AI_OD_ST_YOLOX_PP_S_ANCHORS;
	params->grid_width_M = AI_OD_ST_YOLOX_PP_M_GRID_WIDTH;
	params->grid_height_M = AI_OD_ST_YOLOX_PP_M_GRID_HEIGHT;
	params->pAnchors_M = AI_OD_ST_YOLOX_PP_M_ANCHORS;
	params->grid_width_L = AI_OD_ST_YOLOX_PP_L_GRID_WIDTH;
	params->grid_height_L = AI_OD_ST_YOLOX_PP_L_GRID_HEIGHT;
	params->pAnchors_L = AI_OD_ST_YOLOX_PP_L_ANCHORS;
	params->max_boxes_limit = AI_OD_ST_YOLOX_PP_MAX_BOXES_LIMIT;
	params->conf_threshold = AI_OD_ST_YOLOX_PP_CONF_THRESHOLD;
	params->iou_threshold = AI_OD_ST_YOLOX_PP_IOU_THRESHOLD;

	ret = od_st_yolox_pp_reset(params);
	__ASSERT_NO_MSG(ret == 0);
}

static void model_pp_run(stai_ptr nn_outputs[3], od_pp_out_t *pp_out, od_st_yolox_pp_static_param_t *params)
{
	od_st_yolox_pp_in_t pp_input;
	int ret;

	pp_input.pRaw_detections_S = (float32_t *) nn_outputs[output_order_index[0]];
	pp_input.pRaw_detections_M = (float32_t *) nn_outputs[output_order_index[1]];
	pp_input.pRaw_detections_L = (float32_t *) nn_outputs[output_order_index[2]];
	pp_out->pOutBuff = pp_detections_buffer;
	params->nb_detect = 0;
	ret = od_st_yolox_pp_process(&pp_input, pp_out, params);
	__ASSERT_NO_MSG(ret == 0);
}

void model_thread_ep(void *arg1, void *arg2, void *arg3)
{
	const struct device *const camera_aux_dev = arg1;
	stai_size nn_out_len[STAI_NETWORK_OUT_NUM];
	stai_ptr outputs[STAI_NETWORK_OUT_NUM];
	od_st_yolox_pp_static_param_t pp_params;
	struct video_buffer *vbuf;
	stai_network_info info;
	od_pp_out_t pp_output;
	stai_size output_nb;
	stai_ptr inputs[1];
	uint32_t tick;
	int ret;
	int i;

	ret = k_mutex_init(&boxes_lock);
	__ASSERT_NO_MSG(ret == 0);
	detections_nb = 0;

	model_npu_init();

	/* initialize runtime */
	ret = stai_runtime_init();
	__ASSERT_NO_MSG(ret == STAI_SUCCESS);
	/* init model instance */
	ret = stai_network_init(network_ctx);
	__ASSERT_NO_MSG(ret == STAI_SUCCESS);

	/* setup buffers size */
	ret = stai_network_get_info(network_ctx, &info);
	__ASSERT_NO_MSG(ret == STAI_SUCCESS);
	__ASSERT_NO_MSG(info.n_inputs == 1);
	__ASSERT_NO_MSG(info.n_outputs == 3);
	ret = stai_network_get_outputs(network_ctx, outputs, &output_nb);
	__ASSERT_NO_MSG(ret == STAI_SUCCESS);
	for (i = 0; i < output_nb; i++)
		nn_out_len[i] = info.outputs[i].size_bytes;

	/* pp init */
	model_pp_init(&pp_params, &info);

	while (1) {
		ret = video_dequeue(camera_aux_dev, &vbuf, K_FOREVER);
		__ASSERT_NO_MSG(ret == 0);

		/* run inference */
		 /* setup input buffer. No need cache ops since full hw path DCMIPP -> NPU */
		inputs[0] = vbuf->buffer;
		ret = stai_network_set_inputs(network_ctx, inputs, ARRAY_SIZE(inputs));
		__ASSERT_NO_MSG(ret == STAI_SUCCESS);
		 /* setup output. Invalidate ouputs so post-process access latest inference result */
		for (i = 0; i < output_nb; i++) {
			ret = sys_cache_data_invd_range(outputs[i], nn_out_len[i]);
			__ASSERT_NO_MSG(ret == 0);
		}
		 /* let's go */
		tick = HAL_GetTick();
		Run_Inference(network_ctx);
		tick = HAL_GetTick() - tick;
		latest_inference_time = tick;

		ret = video_enqueue(camera_aux_dev, vbuf);
		__ASSERT_NO_MSG(ret == 0);

		/* post process */
		model_pp_run(outputs, &pp_output, &pp_params);

		/* boxes state */
		ret = k_mutex_lock(&boxes_lock, K_FOREVER);
		__ASSERT_NO_MSG(ret == 0);

		detections_nb = pp_output.nb_detect;
		for (i = 0; i < pp_output.nb_detect; i++)
			detections[i] = pp_output.pOutBuff[i];

		ret = k_mutex_unlock(&boxes_lock);
		__ASSERT_NO_MSG(ret == 0);
	}
}

int model_get_boxes(struct dbox *boxes, int boxes_nb)
{
	int nb = 0;
	int ret;
	int i;

	/* not yet init */
	if (detections_nb < 0)
		return 0;

	ret = k_mutex_lock(&boxes_lock, K_FOREVER);
	__ASSERT_NO_MSG(ret == 0);

	nb = MIN(boxes_nb, detections_nb);
	for (i = 0; i < nb; i++)
		model_detection_to_box(&detections[i], &boxes[i]);

	ret = k_mutex_unlock(&boxes_lock);
	__ASSERT_NO_MSG(ret == 0);

	return nb;
}

int model_get_latest_inference_time()
{
	return latest_inference_time;
}
