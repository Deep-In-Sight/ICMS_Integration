#include "faceID.hpp"
EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);
EA_MEASURE_TIME_DECLARE();

void FaceID::Init()
{
	detectionFlag = false;
}

void FaceID::build(Parser &PARSER)
{
	EA_LOG_SET_LOCAL(PARSER.FACEID_log_level);
	EA_LOG_NOTICE("@@@@@@@@@@@@@@@   FaceID model init @@@@@@@@@@@@@@@         	\n");

	const char *cstr = PARSER.FACEID_ENGINE.c_str();
	
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

	size_t crop_shape[4];
	crop_shape[0] = 1;
	crop_shape[1] = 3;
	crop_shape[2] = 480;
	crop_shape[3] = 720;	

	rgb_LAND = ea_tensor_new(EA_U8, crop_shape, 0);


	size_t grey_shape[4];

	grey_shape[0] = 1;
	grey_shape[1] = 1;
	grey_shape[2] = 480;
	grey_shape[3] = 720;	


		grey_LAND = ea_tensor_new(EA_U8, grey_shape, 0);
}
void FaceID::doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t *ROI_land_crop)
{
	detectionFlag = true;

	EA_MEASURE_TIME_START();	
	ea_net_update_output(model_config.net, model_config.out[0].tensor_name, model_config.out[0].out); // net의 아웃풋과 tensor_name이 같은지 확인하고 net의 내부출력텐서를 out 외부출력텐서로 변경한다.//신경망의 출력값을 저장하는 기본적인 텐서(내부 출력 텐서)를, 사용자가 별도로 제공하는 다른 텐서(외부에서 제공된 텐서)로 변경 //printf("ea_tensor_shape(tmp->out[0].out)[0]  %d \n",ea_tensor_shape(tmp->out[0].out)[0~3]) ;왜 계속 변경하지?UMU
	ea_cvt_color_resize(*hold_image_tensor_pointer, rgb_LAND, EA_COLOR_YUV2RGB_NV12, EA_VP);
	ea_crop_resize(&rgb_LAND, 1, &model_config.in_info[0].tensor, 1, ROI_land_crop, EA_TENSOR_COLOR_MODE_RGB, EA_VP);
	EA_LOG_DEBUG("faceID pre: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();
	ea_net_forward(model_config.net, 1);
	EA_LOG_DEBUG("faceID net: %10ld us \n", ea_gettime_us() - ea_mt_start);


	EA_MEASURE_TIME_START();
	ea_tensor_sync_cache(model_config.out[0].out, EA_VP, EA_CPU);
	EA_LOG_DEBUG("faceID post: %10ld us \n", ea_gettime_us() - ea_mt_start);


	faceid_output_tensor = (float *)ea_tensor_data(model_config.out[0].out);
}

void FaceID::drawLandmark(ea_display_t *display, ea_roi_t *roi)
{

	if (display != NULL)
	{
		for (int i = 0; i < 512; i++)
		{
			EA_LOG_VERBOSE("faceid_output_tensor %d %f \n", i, faceid_output_tensor[i]);
		}
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////


