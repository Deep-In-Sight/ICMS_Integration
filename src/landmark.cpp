#include "landmark.hpp"

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);
EA_MEASURE_TIME_DECLARE();

void Landmark::Init()
{
	for (float &element : landmark_img)
	{
		element = -1;
	}
	detectionFlag = false;
}

void Landmark::build(Parser &PARSER)
{
	EA_LOG_SET_LOCAL(PARSER.LANDMARK_log_level);
	EA_LOG_NOTICE("@@@@@@@@@@@@@@@   LAND model init   @@@@@@@@@@@@@@@         	\n");

	const char* cstr = PARSER.LANDMARK_ENGINE.c_str();

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
	size_t crop_shape[4];
	crop_shape[0] = 1;
	crop_shape[1] = 3;
	crop_shape[2] = 480;
	crop_shape[3] = 720;

	rgb_LAND = ea_tensor_new(EA_U8, crop_shape, 0);
}
void Landmark::doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t *ROI_land_crop)
{
	detectionFlag = true;

	EA_MEASURE_TIME_START();
	ea_net_update_output(model_config.net, model_config.out[0].tensor_name, model_config.out[0].out); // net의 아웃풋과 tensor_name이 같은지 확인하고 net의 내부출력텐서를 out 외부출력텐서로 변경한다.//신경망의 출력값을 저장하는 기본적인 텐서(내부 출력 텐서)를, 사용자가 별도로 제공하는 다른 텐서(외부에서 제공된 텐서)로 변경 //printf("ea_tensor_shape(tmp->out[0].out)[0]  %d \n",ea_tensor_shape(tmp->out[0].out)[0~3]) ;왜 계속 변경하지?UMU
	ea_cvt_color_resize(*hold_image_tensor_pointer, rgb_LAND, EA_COLOR_YUV2RGB_NV12, EA_VP);
	ea_crop_resize(&rgb_LAND, 1, &model_config.in_info[0].tensor, 1, ROI_land_crop, EA_TENSOR_COLOR_MODE_RGB, EA_VP);
	EA_LOG_DEBUG("LAND pre: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();

	ea_net_forward(model_config.net, 1);
	EA_LOG_DEBUG("LAND net: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();

	ea_tensor_sync_cache(model_config.out[0].out, EA_VP, EA_CPU);
	landmark_output_tensor = (float *)ea_tensor_data(model_config.out[0].out);
	postProcess(ROI_land_crop);
	EA_LOG_DEBUG("LAND post: %10ld us \n", ea_gettime_us() - ea_mt_start);




}

void Landmark::postProcess(ea_roi_t *ROI_land_crop)
{
	for (put_i = 0; put_i < 68; put_i++)
	{
		landmark_img[put_i * 2] = ((float)ROI_land_crop->x + landmark_output_tensor[put_i * 2] * (float)ROI_land_crop->w) / 720.0;
		landmark_img[put_i * 2 + 1] = ((float)ROI_land_crop->y + landmark_output_tensor[put_i * 2 + 1] * (float)ROI_land_crop->h) / 480.0;
	}

	if (EA_LOG_GET_LOCAL() == 4)
		for (put_i = 0; put_i < 68; put_i++)
			EA_LOG_VERBOSE("landmark point_%d  x: %f, y: %f \n", put_i, landmark_img[put_i * 2] , landmark_img[put_i * 2 + 1]);
	
}

void Landmark::drawLandmark(ea_display_t *display, ea_roi_t *roi)
{
	if (display != NULL)
	{
		ea_display_obj_params(display)->border_thickness = 10;

		ea_display_obj_params(display)->box_color = EA_16_COLORS_WHITE;

		for (int i = 0; i < 68; i++)
		{
			ea_display_set_bbox(display, NULL, landmark_img[i*2], landmark_img[i*2+1] , 0.001, 0.001);
		}
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
