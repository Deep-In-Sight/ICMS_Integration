#ifndef __facehpp__
#define __facehpp__

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <signal.h>
#include <getopt.h>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <eazyai.h>
#include <opencv2/opencv.hpp>


#include "ICMSparser.hpp"
#include <opencv2/opencv.hpp>
#include "model_management.hpp"



#define SIGMOID(x) (1.0 / (1.0 + exp(-(x))))
#define max_yolo(a, b) (((a) > (b)) ? (a) : (b))
#define min_yolo(a, b) (((a) < (b)) ? (a) : (b))
#define YOLOX_STRIDE_NUM	(3)
#define YOLOX_STRIDE_1		(8)
#define YOLOX_STRIDE_2		(16)
#define YOLOX_STRIDE_3		(32)
#define MAX_OUT_NUM 256
#define MAX_STR_LEN 256
#define MAX_LABEL_LEN 256

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Face : public DLmodel_struct
{

public:

	typedef struct bbox_result_s {
		float score;
		int id;
		char label[MAX_STR_LEN + 7];
		float x_start;
		float y_start;
		float x_end;
		float y_end;
	} bbox_result_t;

	struct face_tracking {
		float x_start;
		float y_start;
		float x_end;
		float y_end;
		bool isValid;
	};

	face_tracking tracked_face; 

	typedef struct detection_result_s {
		bbox_result_t detections[256];
		int valid_det_count;
	} detection_result_t;



	typedef struct nnl_yolox_arm_cfg_s
	{
		char output_name[MAX_STR_LEN];
		float nms_threshold;
		float conf_threshold;
		int class_agnostic;

		int strides[YOLOX_STRIDE_NUM];
		int hw[YOLOX_STRIDE_NUM * 2];
	} nnl_yolox_arm_cfg_t;

	char label[MAX_STR_LEN];

	model_config_t model_config;
	ea_tensor_t *rgb_LAND = NULL;


	detection_result_t *res = NULL;
	nnl_yolox_arm_cfg_t yolox_cfg ;

	void Init();
	void build(Parser &PARSER);

	void doInference(ea_tensor_t **hold_image_tensor_pointer,ea_roi_t* face_box);
	void class_post_process(float* vp_output, char* labels, int label_count,ea_roi_t* face_box);
	void draw_detection_bbox_textbox(ea_display_t *ctx);
	float tracking_iou_v(face_tracking a, bbox_result_s b);
	void tracking();

	bool detectionFlag;
	ea_roi_t ROI_face_crop;

	float conf_threshold;
	float nms_threshold;
	float C_WIDTH;

	int is_driver_valid;

};

#endif /*__facehpp__*/
