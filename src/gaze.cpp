#include "gaze.hpp"
#include <random>

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);
EA_MEASURE_TIME_DECLARE();

void Gaze::Init()
{
	icc_detect_gaze = {-1, -1};

	icc_detect_R_eye_gaze_conf = -1;
	icc_detect_L_eye_gaze_conf	=-1;
	icc_detect_gaze				= {-1, -1};
	icc_detect_R_eye_gaze		= {-1, -1};
	icc_detect_L_eye_gaze		= {-1, -1};
	icc_input_camera			= {-1, -1};
	icc_detect_R_eye_gaze_cal	= {-1, -1};
	icc_detect_L_eye_gaze_cal	= {-1, -1};
	icc_detect_gaze_cal			= {-1, -1};
	icc_detect_eyes_on_road		= -1;



}

void Gaze::build(Parser &PARSER)
{
	EA_LOG_SET_LOCAL(PARSER.GAZE_log_level);
	EA_LOG_NOTICE("@@@@@@@@@@@@@@@   Gaze model init   @@@@@@@@@@         	\n");

	const char *cstr = PARSER.GAZE_ENGINE.c_str();

	icc_input_camera.pitch = PARSER.icc_input_camera_pitch;
	icc_input_camera.yaw = PARSER.icc_input_camera_yaw;

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




}

void Gaze::doInference(ea_tensor_t **hold_image_tensor_pointer, ea_roi_t ROI_face_xywh,int *landmark_img)
{
	ROI_face_xywh.x=ROI_face_xywh.x-240;

	EA_MEASURE_TIME_START();
	ea_net_update_output(model_config.net, model_config.out[0].tensor_name, model_config.out[0].out); // net의 아웃풋과 tensor_name이 같은지 확인하고 net의 내부출력텐서를 out 외부출력텐서로 변경한다.//신경망의 출력값을 저장하는 기본적인 텐서(내부 출력 텐서)를, 사용자가 별도로 제공하는 다른 텐서(외부에서 제공된 텐서)로 변경 //printf("ea_tensor_shape(tmp->out[0].out)[0]  %d \n",ea_tensor_shape(tmp->out[0].out)[0~3]) ;왜 계속 변경하지?UMU
	ea_crop_resize(hold_image_tensor_pointer, 1, &model_config.in_info[0].tensor, 1, &ROI_face_xywh, EA_TENSOR_COLOR_MODE_RGB, EA_VP);
	EA_LOG_DEBUG("gaze pre: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();
	ea_net_forward(model_config.net, 1);
	EA_LOG_DEBUG("gaze net: %10ld us \n", ea_gettime_us() - ea_mt_start);

	EA_MEASURE_TIME_START();
	ea_tensor_sync_cache(model_config.out[0].out, EA_VP, EA_CPU);
	gaze_net_out = std::vector<float>((float *)ea_tensor_data(model_config.out[0].out), (float *)ea_tensor_data(model_config.out[0].out) + 180);
	postProcess();
	EA_LOG_DEBUG("gaze post: %10ld us \n", ea_gettime_us() - ea_mt_start);

	// for (int put_i = 0; put_i < 68; put_i++)
	// {
	// 	std::cout<<"landmark_img "<< put_i << "  " << landmark_img[put_i*2]<<"  "<<landmark_img[put_i*2+1]<<std::endl;
	// }

}




std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    float max_elem = *std::max_element(x.begin(), x.end());
    float sum = 0.0;

    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = exp(x[i] - max_elem);
        sum += result[i];
    }

    for (size_t i = 0; i < x.size(); ++i) {
        result[i] /= sum;
    }

    return result;
}
float custom_inner_product(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}


void Gaze::postProcess(){

    std::vector<float> gaze_pitch(gaze_net_out.begin(), gaze_net_out.begin() + 90);
    std::vector<float> gaze_yaw(gaze_net_out.begin() + 90, gaze_net_out.begin()+180);


    gaze_pitch = softmax(gaze_pitch);
    gaze_yaw = softmax(gaze_yaw);

    std::vector<float> idx_tensor(90);
    for (size_t i = 0; i < 90; ++i) {
        idx_tensor[i] = static_cast<float>(i);
    }

    float pitch_predicted = custom_inner_product(gaze_pitch, idx_tensor) * 4.0 - 180.0;
    float yaw_predicted = (custom_inner_product(gaze_yaw, idx_tensor) * 4.0 - 180.0)*-1.0;
	icc_detect_gaze.pitch=yaw_predicted;
	icc_detect_gaze.yaw=pitch_predicted;
}
