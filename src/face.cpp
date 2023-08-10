#include "face.hpp"

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);
EA_MEASURE_TIME_DECLARE();

void Face::Init()
{
	is_driver_valid = 0;
}

void Face::build(Parser &PARSER)
{
	EA_LOG_SET_LOCAL(PARSER.YOLO_log_level);
	EA_LOG_NOTICE("\n@@@@@@@@@@@@@@@   FACE model init   @@@@@@@@@@@@@@@         	\n");

	conf_threshold = PARSER.YOLO_CONFIDENCE;
	nms_threshold = PARSER.YOLO_NMS;
	C_WIDTH = PARSER.YOLO_WIDTH;

	std::strcpy(label, "face");

	const char *cstr = PARSER.YOLO_ENGINE.c_str();

	const size_t *shape = NULL;
	const size_t *shape_in = NULL;

	ea_net_params_t net_param2;
	memset(&net_param2, 0, sizeof(net_param2));
	model_config.net = ea_net_new(&net_param2); // Create a network
	ea_net_load(model_config.net, EA_NET_LOAD_FILE, (void *)cstr, 1);
	model_config.in_num = ea_net_input_num(model_config.net); // 1
	model_config.in_info[0].name = ea_net_input_name(model_config.net, 0);
	model_config.in_info[0].tensor = ea_net_input_by_index(model_config.net, 0);
	model_config.out_num = ea_net_output_num(model_config.net); // 1
	model_config.out_info[0].name = ea_net_output_name(model_config.net, 0);
	model_config.out_info[0].tensor = ea_net_output_by_index(model_config.net, 0);
	int nn_out_num = model_config.out_num; // 1
	model_config.out = (out_info_t *)malloc(sizeof(out_info_t) * nn_out_num);
	shape_in = ea_tensor_shape(model_config.in_info[0].tensor);
	shape = ea_tensor_shape(model_config.out_info[0].tensor);
	model_config.out[0].tensor_name = model_config.out_info[0].name;
	// ea_tensor_new: cavalry 메모리에서 할당된 버퍼로 텐서를 만듬. 저장된 cavalry 메모리에서 텐서 데이터는 CPU와 VP가 공유함
	model_config.out[0].out = ea_tensor_new(ea_tensor_dtype(model_config.out_info[0].tensor), shape, ea_tensor_pitch(model_config.out_info[0].tensor)); // EA_F32 | x | 32
	model_config.out_num = nn_out_num;

	EA_LOG_NOTICE("network input: %s\n", model_config.in_info[0].name);
	EA_LOG_NOTICE("network output: %s\n", model_config.out_info[0].name);
	EA_LOG_NOTICE("in_num            	%d \n", model_config.in_num);
	EA_LOG_NOTICE("in_info[0].name  	%s \n", model_config.in_info[0].name);
	EA_LOG_NOTICE("in_info[0].tensor	%p \n", model_config.in_info[0].tensor);
	EA_LOG_NOTICE("out_num           	%d \n", model_config.out_num);
	EA_LOG_NOTICE("out_info[0].name  	%s \n", model_config.out_info[0].name);
	EA_LOG_NOTICE("out_info[0].tensor	%p \n", model_config.out_info[0].tensor);
	EA_LOG_NOTICE("input  shape[N:%d]  shape[C:%d]  shape[E:%d]  shape[E:%d]  \n", shape_in[EA_N], shape_in[EA_C], shape_in[EA_H], shape_in[EA_W]);
	EA_LOG_NOTICE("output shape[N:%d]  shape[C:%d]  shape[E:%d]  shape[E:%d]  \n", shape[EA_N], shape[EA_C], shape[EA_H], shape[EA_W]);
	EA_LOG_NOTICE("ea_tensor_dtype(out_info[0].tensor) %d \n", ea_tensor_dtype(model_config.out_info[0].tensor));
	EA_LOG_NOTICE("ea_tensor_pitch(out_info[0].tensor %d \n", ea_tensor_pitch(model_config.out_info[0].tensor));
	EA_LOG_NOTICE("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@         	\n\n");

	// custom build
	yolox_cfg.strides[0] = YOLOX_STRIDE_1;
	yolox_cfg.strides[1] = YOLOX_STRIDE_2;
	yolox_cfg.strides[2] = YOLOX_STRIDE_3;
	for (int i = 0; i < YOLOX_STRIDE_NUM; i++)
	{
		yolox_cfg.hw[2 * i + 0] = ea_tensor_shape(model_config.in_info[0].tensor)[EA_H] / yolox_cfg.strides[i]; // height
		yolox_cfg.hw[2 * i + 1] = ea_tensor_shape(model_config.in_info[0].tensor)[EA_W] / yolox_cfg.strides[i]; // width
	}
	res = (detection_result_t *)malloc(sizeof(detection_result_t));

	size_t crop_shape[4];
	crop_shape[0] = 1;
	crop_shape[1] = 3;
	crop_shape[2] = 480;
	crop_shape[3] = 720;
	rgb_LAND = ea_tensor_new(EA_U8, crop_shape, 0);

	ROI_face_crop.x = (720 - C_WIDTH) / 2;
	ROI_face_crop.y = 0;
	ROI_face_crop.w = C_WIDTH;
	ROI_face_crop.h = 480;

	tracked_face.x_start = 0;
	tracked_face.x_end = 0;
	tracked_face.y_start = 0;
	tracked_face.y_end = 0;
	tracked_face.isValid = 0;
}
void Face::doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t *face_box)
{

	EA_MEASURE_TIME_START();
	ea_net_update_output(model_config.net, model_config.out[0].tensor_name, model_config.out[0].out); // net의 아웃풋과 tensor_name이 같은지 확인하고 net의 내부출력텐서를 out 외부출력텐서로 변경한다.//신경망의 출력값을 저장하는 기본적인 텐서(내부 출력 텐서)를, 사용자가 별도로 제공하는 다른 텐서(외부에서 제공된 텐서)로 변경 //printf("ea_tensor_shape(tmp->out[0].out)[0]  %d \n",ea_tensor_shape(tmp->out[0].out)[0~3]) ;왜 계속 변경하지?UMU
	ea_cvt_color_resize(*hold_image_tensor_pointer, rgb_LAND, EA_COLOR_YUV2RGB_NV12, EA_VP);
	ea_crop_resize(&rgb_LAND, 1, &model_config.in_info[0].tensor, 1, &ROI_face_crop, EA_TENSOR_COLOR_MODE_RGB, EA_VP);
	EA_LOG_DEBUG("Face pre: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();
	ea_net_forward(model_config.net, 1);
	EA_LOG_DEBUG("Face net: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();
	ea_tensor_sync_cache(model_config.out[0].out, EA_VP, EA_CPU); // Read the inference result
	class_post_process((float *)ea_tensor_data(model_config.out[0].out), label, 1, face_box);
	EA_LOG_DEBUG("Face post: %10ld us \n", ea_gettime_us() - ea_mt_start);
}

void Face::class_post_process(float *vp_output, char *labels, int label_count, ea_roi_t *face_box)
{

	int rval = EA_SUCCESS;
	int class_num = 1;
	int max_det_num = 2100;
	size_t pitch = 32;
	float *p_next_data = NULL;
	float *p_box = NULL;
	float obj_conf;
	float *p_class_conf = NULL;
	int height, width, stride;
	float class_conf;
	int class_pred;
	float **valid_x1y1x2y2score = NULL;
	int *valid_count = NULL;
	float *class_x1y1x2y2score = NULL;
	int class_valid_count;
	float **nms_x1y1x2y2score = NULL;
	int *nms_valid_count = NULL;
	int i, s, c, h, w;
	ea_tensor_t *output_tensor = NULL;
	do
	{

		valid_x1y1x2y2score = (float **)malloc(class_num * sizeof(float *));
		RVAL_ASSERT(valid_x1y1x2y2score != NULL);
		valid_count = (int *)malloc(class_num * sizeof(int));
		RVAL_ASSERT(valid_count != NULL);
		memset(valid_count, 0, class_num * sizeof(int));
		for (c = 0; c < class_num; c++)
		{
			valid_x1y1x2y2score[c] = (float *)malloc(max_det_num * 5 * sizeof(float));
			RVAL_ASSERT(valid_x1y1x2y2score[c] != NULL);
		}
		RVAL_BREAK();

		p_next_data = vp_output;
		for (s = 0; s < YOLOX_STRIDE_NUM; s++)
		{
			stride = yolox_cfg.strides[s];
			height = yolox_cfg.hw[2 * s + 0];
			width = yolox_cfg.hw[2 * s + 1];
			for (h = 0; h < height; h++)
			{
				for (w = 0; w < width; w++)
				{
					p_box = p_next_data;
					obj_conf = p_next_data[4];
					if (obj_conf >= conf_threshold /*yolox_cfg->conf_threshold*/)
					{
						p_class_conf = p_next_data + 5;
						class_conf = p_class_conf[0];
						class_pred = 0;
						for (c = 1; c < class_num; c++)
						{
							if (class_conf < p_class_conf[c])
							{
								class_conf = p_class_conf[c];
								class_pred = c;
							}
						}

						if (class_conf * obj_conf >= conf_threshold /*yolox_cfg->conf_threshold*/)
						{
							class_x1y1x2y2score = valid_x1y1x2y2score[class_pred];
							class_valid_count = valid_count[class_pred];
							class_x1y1x2y2score[class_valid_count * 5 + 4] = class_conf * obj_conf;
							class_x1y1x2y2score[class_valid_count * 5 + 0] = (p_box[0] + w) * stride;								// x center
							class_x1y1x2y2score[class_valid_count * 5 + 1] = (p_box[1] + h) * stride;								// y center
							class_x1y1x2y2score[class_valid_count * 5 + 2] = exp(p_box[2]) * stride;								// w
							class_x1y1x2y2score[class_valid_count * 5 + 3] = exp(p_box[3]) * stride;								// h
							class_x1y1x2y2score[class_valid_count * 5 + 0] -= class_x1y1x2y2score[class_valid_count * 5 + 2] / 2.0; // x start
							class_x1y1x2y2score[class_valid_count * 5 + 1] -= class_x1y1x2y2score[class_valid_count * 5 + 3] / 2.0; // y start
							class_x1y1x2y2score[class_valid_count * 5 + 2] += class_x1y1x2y2score[class_valid_count * 5 + 0];		// x end
							class_x1y1x2y2score[class_valid_count * 5 + 3] += class_x1y1x2y2score[class_valid_count * 5 + 1];		// y end
							EA_LOG_DEBUG("x1y1x2y2: %f %f %f %f %f\n", class_x1y1x2y2score[class_valid_count * 5 + 0],
										 class_x1y1x2y2score[class_valid_count * 5 + 1], class_x1y1x2y2score[class_valid_count * 5 + 2],
										 class_x1y1x2y2score[class_valid_count * 5 + 3], class_x1y1x2y2score[class_valid_count * 5 + 4]);
							valid_count[class_pred]++;
						}
					}

					p_next_data = (float *)((uint8_t *)p_next_data + pitch);
				}
			}
		}

		nms_x1y1x2y2score = (float **)malloc(class_num * sizeof(float *));
		RVAL_ASSERT(nms_x1y1x2y2score != NULL);
		nms_valid_count = (int *)malloc(class_num * sizeof(int));
		RVAL_ASSERT(nms_valid_count != NULL);
		memset(nms_valid_count, 0, class_num * sizeof(int));
		for (c = 0; c < class_num; c++)
		{
			nms_x1y1x2y2score[c] = (float *)malloc(valid_count[c] * 5 * sizeof(float));
			RVAL_ASSERT(nms_x1y1x2y2score[c] != NULL);
		}
		RVAL_BREAK();

		for (c = 0; c < class_num; c++)
		{
			RVAL_OK(ea_nms(valid_x1y1x2y2score[c], NULL, 0, valid_count[c], nms_threshold /*yolox_cfg->nms_threshold*/, 0 /*use_iou_min*/, 0 /*tok_k*/,
						   nms_x1y1x2y2score[c], NULL, &nms_valid_count[c]));
		}
		RVAL_BREAK();

		res->valid_det_count = 0;

		for (c = 0; c < class_num; c++)
		{
			for (i = 0; i < nms_valid_count[c]; i++)
			{

				res->detections[res->valid_det_count].id = c;
				res->detections[res->valid_det_count].score = nms_x1y1x2y2score[c][i * 5 + 4];
				res->detections[res->valid_det_count].x_start = nms_x1y1x2y2score[c][i * 5 + 0] / ea_tensor_shape(model_config.in_info[0].tensor)[EA_W];
				if (res->detections[res->valid_det_count].x_start < 0.02)
					continue;
				res->detections[res->valid_det_count].x_start = max_yolo(0.0, res->detections[res->valid_det_count].x_start);
				res->detections[res->valid_det_count].x_start = min_yolo(1.0, res->detections[res->valid_det_count].x_start);
				res->detections[res->valid_det_count].x_start = (res->detections[res->valid_det_count].x_start * (C_WIDTH / 720.0)) + ((720 - C_WIDTH) / 2.0 / 720.0);

				res->detections[res->valid_det_count].y_start = nms_x1y1x2y2score[c][i * 5 + 1] / ea_tensor_shape(model_config.in_info[0].tensor)[EA_H];
				if (res->detections[res->valid_det_count].y_start < 0.02)
					continue;
				res->detections[res->valid_det_count].y_start = max_yolo(0.0, res->detections[res->valid_det_count].y_start);
				res->detections[res->valid_det_count].y_start = min_yolo(1.0, res->detections[res->valid_det_count].y_start);

				res->detections[res->valid_det_count].x_end = nms_x1y1x2y2score[c][i * 5 + 2] / ea_tensor_shape(model_config.in_info[0].tensor)[EA_W];
				if (res->detections[res->valid_det_count].x_end > 0.98)
					continue;
				res->detections[res->valid_det_count].x_end = max_yolo(res->detections[res->valid_det_count].x_start, res->detections[res->valid_det_count].x_end);
				res->detections[res->valid_det_count].x_end = min_yolo(1.0, res->detections[res->valid_det_count].x_end);
				res->detections[res->valid_det_count].x_end = (res->detections[res->valid_det_count].x_end * (C_WIDTH / 720.0)) + ((720 - C_WIDTH) / 2.0 / 720.0);

				res->detections[res->valid_det_count].y_end = nms_x1y1x2y2score[c][i * 5 + 3] / ea_tensor_shape(model_config.in_info[0].tensor)[EA_H];
				if (res->detections[res->valid_det_count].y_end > 0.98)
					continue;
				res->detections[res->valid_det_count].y_end = max_yolo(res->detections[res->valid_det_count].y_start, res->detections[res->valid_det_count].y_end);
				res->detections[res->valid_det_count].y_end = min_yolo(1.0, res->detections[res->valid_det_count].y_end);

				memset(res->detections[res->valid_det_count].label, 0, sizeof(res->detections[res->valid_det_count].label));
				if (labels != NULL)
				{
					RVAL_ASSERT(res->detections[res->valid_det_count].id < label_count);
					snprintf(res->detections[res->valid_det_count].label, sizeof(res->detections[res->valid_det_count].label),
							 "%s", labels);
				}

				EA_LOG_DEBUG("det %d: %f %f %f %f %f\n", i,
							 nms_x1y1x2y2score[c][i * 5 + 0], nms_x1y1x2y2score[c][i * 5 + 1],
							 nms_x1y1x2y2score[c][i * 5 + 2], nms_x1y1x2y2score[c][i * 5 + 3],
							 nms_x1y1x2y2score[c][i * 5 + 4]);
				EA_LOG_DEBUG("det %d: %f %f %f %f %f\n", i,
							 res->detections[res->valid_det_count].x_start, res->detections[res->valid_det_count].y_start,
							 res->detections[res->valid_det_count].x_end, res->detections[res->valid_det_count].y_end,
							 res->detections[res->valid_det_count].score);

				res->valid_det_count++;

				if (res->valid_det_count >= MAX_OUT_NUM)
				{
					break;
				}
			}

			if (res->valid_det_count >= MAX_OUT_NUM)
			{
				EA_LOG_NOTICE("It reaches max detection number %d. Please concider to increase MAX_OUT_NUM\n", MAX_OUT_NUM);
				break;
			}
		}

		tracking();
		face_box->x = (int)(tracked_face.x_start * 720);
		face_box->y = (int)(tracked_face.y_start * 480);
		face_box->w = (int)((tracked_face.x_end - tracked_face.x_start) * 720);
		face_box->h = (int)((tracked_face.y_end - tracked_face.y_start) * 480);

	} while (0);

	if (res->valid_det_count != 0)
		is_driver_valid = 1;
	else
	{
		tracked_face.x_start = -1;
		tracked_face.y_start = -1;
		tracked_face.x_end = -1;
		tracked_face.y_end = -1;
		tracked_face.isValid = 0;
		is_driver_valid = 0;

	}

	if (valid_x1y1x2y2score)
	{
		for (c = 0; c < class_num; c++)
		{
			free(valid_x1y1x2y2score[c]);
		}

		free(valid_x1y1x2y2score);
	}

	if (valid_count)
	{
		free(valid_count);
	}

	if (nms_x1y1x2y2score)
	{
		for (c = 0; c < class_num; c++)
		{
			free(nms_x1y1x2y2score[c]);
		}

		free(nms_x1y1x2y2score);
	}

	if (nms_valid_count)
	{
		free(nms_valid_count);
	}
}

void Face::draw_detection_bbox_textbox(ea_display_t *display)
{
	int rval = 0;
	char text[MAX_LABEL_LEN];
	int i;
	do
	{
		if (display != NULL)
		{
			ea_display_obj_params(display)->obj_win_w = 1.0;
			ea_display_obj_params(display)->obj_win_h = 1.0;

			ea_display_obj_params(display)->border_thickness = 5;
			ea_display_obj_params(display)->font_size = 32;
			ea_display_obj_params(display)->text_color = EA_16_COLORS_WHITE;

			RVAL_ASSERT(res != NULL);

			for (i = 0; i < res->valid_det_count; i++)
			{
				ea_display_obj_params(display)->box_color = EA_16_COLORS_WHITE;
				ea_display_set_bbox(display, "FACE", // 0~1 no pixel
									res->detections[i].x_start,
									res->detections[i].y_start,
									res->detections[i].x_end - res->detections[i].x_start,
									res->detections[i].y_end - res->detections[i].y_start);
			}
			if (res->valid_det_count != 0)
			{
				ea_display_obj_params(display)->text_color = EA_16_COLORS_RED;
				ea_display_obj_params(display)->box_color = EA_16_COLORS_RED;

				ea_display_set_bbox(display, "TARGET", // 0~1 no pixel
									tracked_face.x_start,
									tracked_face.y_start,
									tracked_face.x_end - tracked_face.x_start,
									tracked_face.y_end - tracked_face.y_start);
			}
		}
	} while (0);
}

float Face::tracking_iou_v(face_tracking a, bbox_result_s b)
{
	float interArea = std::max(0.0f, std::min(a.x_end, b.x_end) - std::max(a.x_start, b.x_start)) * std::max(0.0f, std::min(a.y_end, b.y_end) - std::max(a.y_start, b.y_start));
	float unionArea = (a.x_end - a.x_start) * (a.y_end - a.y_start) + (b.x_end - b.x_start) * (b.y_end - b.y_start) - interArea;
	return interArea / unionArea;
}

void Face::tracking()
{

	if (res->valid_det_count != 0)
	{
		float maxIou = 0.0f;
		int maxIndex = -1;
		int check_iou = 1;

		int maxScore_else = 0.0f;
		int maxIndex_else = -1;

		if (tracked_face.isValid == 1)
		{
			for (int i = 0; i < res->valid_det_count; ++i)
			{
				float currentIou = tracking_iou_v(tracked_face, res->detections[i]);

				if (currentIou > maxIou)
				{
					maxIou = currentIou;
					maxIndex = i;
				}
			}

			if (maxIndex != -1)
			{
				tracked_face.x_start = res->detections[maxIndex].x_start;
				tracked_face.x_end = res->detections[maxIndex].x_end;
				tracked_face.y_start = res->detections[maxIndex].y_start;
				tracked_face.y_end = res->detections[maxIndex].y_end;
			}
			else
				tracked_face.isValid = false;
		}
		else
		{
			for (int i = 0; i < res->valid_det_count; ++i)
			{
				float currentScore = res->detections[i].score;
				if (currentScore > maxScore_else)
				{
					maxScore_else = currentScore;
					maxIndex_else = i;
				}
			}

			tracked_face.x_start = res->detections[maxIndex_else].x_start;
			tracked_face.x_end = res->detections[maxIndex_else].x_end;
			tracked_face.y_start = res->detections[maxIndex_else].y_start;
			tracked_face.y_end = res->detections[maxIndex_else].y_end;
			tracked_face.isValid = true;
		}
	}
}
