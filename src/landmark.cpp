#include "landmark.hpp"
#include "opencv2/calib3d.hpp"

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);
EA_MEASURE_TIME_DECLARE();

void Landmark::Init()
{

	headpose_history.roll = -1;
	headpose_history.pitch = -1;
	headpose_history.yaw = -1;

	headpose_cal.roll = -1;
	headpose_cal.pitch = -1;
	headpose_cal.yaw = -1;

	nose_result_x = -1;
	nose_result_y = -1;
	ICC_DECISION_DISTRACTION = -1;
	icc_detect_left_eye_open_mm = -1;
	icc_detect_right_eye_open_mm = -1;
	ICC_DECISION_DROWSINESS = -1;
	icc_detect_is_yawning = -1;
	icc_detect_driver_talking_type = -1;
	ICC_DECISION_HEAD_GESTURE_SHAKE = -1;
	ICC_DECISION_HEAD_GESTURE_BOB = -1;
	ICC_DECISION_HEAD_GESTURE_NOD = -1;
	icc_detect_is_face_valid = -1;

	icc_detect_head_gesture=-1;
	for (int &element : landmark_img)
	{
		element = -1;
	}
	detectionFlag = false;
}

void Landmark::build(Parser &PARSER)
{
	EA_LOG_SET_LOCAL(PARSER.LANDMARK_log_level);
	EA_LOG_NOTICE("@@@@@@@@@@@@@@@   LAND model init   @@@@@@@@@@@@@@@         	\n");
	camera_w= PARSER.camera_w;
	camera_h= PARSER.camera_h;
	DISTRACTION_landmark_keep[0] = -1;
	DISTRACTION_landmark_keep[1] = -1;
	DISTRACTION_landmark_keep[2] = -1;
	DISTRACTION_landmark_keep[3] = -1;
	DISTRACTION_landmark_keep[4] = -1;
	DISTRACTION_landmark_keep[5] = -1;
	headpose_cal_value.roll = PARSER.icc_input_camera_roll;
	headpose_cal_value.pitch = PARSER.icc_input_camera_pitch;
	headpose_cal_value.yaw = PARSER.icc_input_camera_yaw;

	const char *cstr = PARSER.LANDMARK_ENGINE.c_str();

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

	is_mouth_open = false;
}
void Landmark::doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t ROI_face_crop, int Frame_num)
{
	detectionFlag = true;
	ROI_face_crop.x=ROI_face_crop.x-240;
	EA_MEASURE_TIME_START();
	ea_net_update_output(model_config.net, model_config.out[0].tensor_name, model_config.out[0].out); // net의 아웃풋과 tensor_name이 같은지 확인하고 net의 내부출력텐서를 out 외부출력텐서로 변경한다.//신경망의 출력값을 저장하는 기본적인 텐서(내부 출력 텐서)를, 사용자가 별도로 제공하는 다른 텐서(외부에서 제공된 텐서)로 변경 //printf("ea_tensor_shape(tmp->out[0].out)[0]  %d \n",ea_tensor_shape(tmp->out[0].out)[0~3]) ;왜 계속 변경하지?UMU
	// ea_cvt_color_resize(*hold_image_tensor_pointer, rgb_LAND, EA_COLOR_YUV2RGB_NV12, EA_VP);
	ea_crop_resize(hold_image_tensor_pointer, 1, &model_config.in_info[0].tensor, 1, &ROI_face_crop, EA_TENSOR_COLOR_MODE_RGB, EA_VP);
	EA_LOG_DEBUG("LAND pre: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();

	ea_net_forward(model_config.net, 1);
	EA_LOG_DEBUG("LAND net: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();

	ea_tensor_sync_cache(model_config.out[0].out, EA_VP, EA_CPU);
	landmark_output_tensor = (float *)ea_tensor_data(model_config.out[0].out);

	postProcess(&ROI_face_crop, Frame_num);
	EA_LOG_DEBUG("LAND post: %10ld us \n", ea_gettime_us() - ea_mt_start);
}

void Landmark::postProcess(ea_roi_t *ROI_face_crop, int Frame_num)
{
	for (put_i = 0; put_i < 68; put_i++)
	{
		landmark_img[put_i * 2] = ROI_face_crop->x + (landmark_output_tensor[put_i * 2] * ROI_face_crop->w);
		landmark_img[put_i * 2 + 1] = ROI_face_crop->y + (landmark_output_tensor[put_i * 2 + 1] * ROI_face_crop->h);
	}
}