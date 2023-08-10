#include "attribute.hpp"

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);
EA_MEASURE_TIME_DECLARE();

void Attribute::Init()
{
	// for(int i = 0; i < 12; i++) 
	// 	attr_output_tensor[i] = -1.0f; // Initialize all elements to -1.0
	MaleFemale=-1;
	mask=-1;
	glasses=-1;
	age_YMO=-1;
	action_SP=-1;


}

void Attribute::build(Parser &PARSER)
{
	EA_LOG_SET_LOCAL(PARSER.ATTR_log_level);
	EA_LOG_NOTICE("@@@@@@@@@@@@@@@   Attribute model init   @@@@@@@@@@         	\n");

	const char *cstr = PARSER.ATTR_ENGINE.c_str();


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

	attr_output_tensor= new float[12];

	MaleFemale=-1;
	mask=-1;
	glasses=-1;
	age_YMO=-1;
	action_SP=-1;



}
void Attribute::doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t *ROI_land_crop)
{
	detectionFlag = true;

	EA_MEASURE_TIME_START();
	ea_net_update_output(model_config.net, model_config.out[0].tensor_name, model_config.out[0].out); // net의 아웃풋과 tensor_name이 같은지 확인하고 net의 내부출력텐서를 out 외부출력텐서로 변경한다.//신경망의 출력값을 저장하는 기본적인 텐서(내부 출력 텐서)를, 사용자가 별도로 제공하는 다른 텐서(외부에서 제공된 텐서)로 변경 //printf("ea_tensor_shape(tmp->out[0].out)[0]  %d \n",ea_tensor_shape(tmp->out[0].out)[0~3]) ;왜 계속 변경하지?UMU
	ea_cvt_color_resize(*hold_image_tensor_pointer, rgb_LAND, EA_COLOR_YUV2RGB_NV12, EA_VP);
	ea_crop_resize(&rgb_LAND, 1, &model_config.in_info[0].tensor, 1, ROI_land_crop, EA_TENSOR_COLOR_MODE_RGB, EA_VP);
	EA_LOG_DEBUG("ATTR pre: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();
	ea_net_forward(model_config.net, 1);
	EA_LOG_DEBUG("ATTR net: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();
	ea_tensor_sync_cache(model_config.out[0].out, EA_VP, EA_CPU);
	EA_LOG_DEBUG("ATTR post: %10ld us \n", ea_gettime_us() - ea_mt_start);

	// attr_output_tensor = (float *)ea_tensor_data(model_config.out[0].out);

	attr_net_out = std::vector<float>((float *)ea_tensor_data(model_config.out[0].out), (float *)ea_tensor_data(model_config.out[0].out) + 12);
	postProcess();
}


void displayHandler(ea_display_t *display,int condition, const char *default_text, std::map<int, const char *> text_map, float &y, float x, float w, float h)
{
	ea_display_obj_params(display)->text_color = (condition == -1) ? EA_16_COLORS_BLUE : EA_16_COLORS_RED;
	ea_display_set_textbox(display, (condition == -1) ? default_text : text_map[condition], x, y, w, h);
	y += 0.03;
}

void Attribute::drawLandmark(ea_display_t *display, ea_roi_t *roi)
{
	std::map<int, const char *> gender_text_map = {{0, "Male"}, {1, "Female"}};
	std::map<int, const char *> mask_text_map = {{0, "Mask_NO"}, {1, "Mask"}};
	std::map<int, const char *> glasses_text_map = {{0, "Glasses_NO"}, {1, "Glasses"}};
	std::map<int, const char *> age_YMO_text_map = {{0, "Young"}, {1, "Middle"}, {2, "Old"}};
	std::map<int, const char *> action_SP_text_map = {{0, "Normal"}, {1, "Smoking"}, {2, "Phone"}};

	
	ea_display_obj_params(display)->font_size = 24;
	ea_display_obj_params(display)->border_thickness = 0;
	ea_display_obj_params(display)->text_color = EA_16_COLORS_AQUA;

	float x = 0.01;
	float y = 0;
	float w = 0.001;
	float h = 0.001;

	displayHandler(display,MaleFemale, "Gender", gender_text_map, y, x, w, h);
	displayHandler(display,mask, "Mask", mask_text_map, y, x, w, h);
	displayHandler(display,glasses, "Glasses", glasses_text_map, y, x, w, h);
	displayHandler(display,age_YMO, "Age", age_YMO_text_map, y, x, w, h);
	displayHandler(display,action_SP, "Action", action_SP_text_map, y, x, w, h);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Attribute::postProcess(){

	if (attr_net_out[0]>0.8)
		MaleFemale=0;
	else if (attr_net_out[1]>0.8)
		MaleFemale=1;
	else
		MaleFemale=-1;

	if (attr_net_out[2]>0.8)
		mask=0;
	else if (attr_net_out[3]>0.8)
		mask=1;
	else
		mask=-1;

	if (attr_net_out[4]>0.8)
		glasses=0;
	else if (attr_net_out[5]>0.8)
		glasses=1;
	else
		glasses=-1;


	if (attr_net_out[6]>0.8)
		age_YMO=0;
	else if (attr_net_out[7]>0.8)
		age_YMO=1;
	else if (attr_net_out[8]>0.8)
		age_YMO=2;
	else
		age_YMO=-1;

	if (attr_net_out[9]>0.8)
		action_SP=0;
	else if (attr_net_out[10]>0.8)
		action_SP=1;
	else if (attr_net_out[11]>0.8)
		action_SP=2;
	else
		action_SP=-1;




}
