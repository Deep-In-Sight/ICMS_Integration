#ifndef __DinsightCallbackStruct__
#define __DinsightCallbackStruct__

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

struct face_result // 0~1
{
	int x_start;
	int y_start;
	int x_end;
	int y_end;
};

struct head_pose
{
	float roll;
	float pitch;
	float yaw;

};

struct icc_detect_head_position_cal_struct
{
	int x;
	int y;
	int z;
};
struct icc_detect_gaze_struct
{
	float pitch;
	float yaw;
};

typedef struct L1CallbackStruct
{
	int icc_detect_is_driver_valid;
	int icc_detect_is_face_valid;
	// int icc_detect_is_face_real;
	int icc_detect_has_glasses;
	int icc_detect_camera_status; //@@
	int icc_detect_steering_wheel_detect;	//@@
	int icc_detect_is_limited_performance; //@@
	int icc_detect_engine_state; //@@
	int icc_detect_nFrame_num;
	int icc_detect_frame_state; //@@
	icc_detect_head_position_cal_struct icc_detect_head_position_cal;
	head_pose icc_detect_head_pose_cal;
	int icc_detect_L_eye_position_cal; //@@
	int icc_detect_R_eye_position_cal; //@@
	float icc_detect_left_eye_open_mm;
	float icc_detect_right_eye_open_mm;
	icc_detect_gaze_struct icc_detect_gaze_cal; //@@ ^^
	int icc_detect_aoi; //@@
	int icc_detect_is_yawning;
	int icc_detect_driver_talking_type;
	int icc_detect_is_day_dreaming; //@@
	int icc_detect_n_yawn_count; //@@ 
	icc_detect_head_position_cal_struct icc_detect_head_position;
	head_pose icc_detect_head_pose;
	int icc_detect_drowsiness; //@@ 
	int icc_detect_distraction; //@@ 
	int icc_detect_last_blink_duration; //@@ 
	int icc_detect_blink_rate; //@@ 
	int icc_detect_distraction_level; //@@ 	^^
	int icc_detect_drowsiness_level; //@@ 	^^
	int icc_detect_time_on_road; //@@ 
	int icc_detect_time_off_road; //@@ 
	int icc_detect_cumulative_time_off_road; //@@ 
	int icc_detect_right_eye_open_perc; //@@ 
	int icc_detect_left_eye_open_perc; //@@ 
	int icc_detect_R_eye_position; //@@ 
	icc_detect_gaze_struct icc_detect_R_eye_gaze; //@@ ^^
	int icc_detect_R_eye_gaze_conf; //@@ ^^
	icc_detect_gaze_struct icc_detect_R_eye_gaze_cal; //@@ ^^
	int icc_detect_L_eye_position; //@@ 
	icc_detect_gaze_struct icc_detect_L_eye_gaze; //@@ ^^
	int icc_detect_L_eye_gaze_conf; //@@ ^^
	icc_detect_gaze_struct icc_detect_L_eye_gaze_cal; //@@ ^^
	icc_detect_gaze_struct icc_detect_gaze; //@@ ^^
	int icc_detect_gaze_xyz; //@@ 
	int icc_detect_gaze_cal_xyz; //@@ 
	face_result icc_detect_head_rect;
	int icc_detect_eyes_on_road; //@@ ^^
	int icc_detect_head_on_road; //@@ 
	int icc_detect_has_mask;
	int icc_detect_is_using_cellphone; 
	int icc_detect_is_smoking;
	int icc_detect_is_eating;  //@@ 
	// int icc_detect_is_drinking;
	int icc_detect_person_id;
	int icc_detect_person_id_matches;   //@@
	int icc_detect_person_id_num; //UMU
	int icc_detect_head_gesture; 
	int icc_detect_dangerous_behavior_detect;   //@@
	int icc_detect_driver_gender_estimate; 
	int icc_detect_driver_age_estimate; 
	int icc_detect_is_wearing_seatbelt;   //@@


};


typedef struct L2CallbackStruct
{


	int ICC_DECISION_IS_DRIVER_VALID;
	// int ICC_DECISION_FACE_LIVENESS;
	int ICC_DECISION_GLASSES;
	int ICC_DECISION_CAMERA_BLOCKAGE; //@@
	int ICC_DECISION_CAMERA_STATE; //@@
	int ICC_DECISION_STEERING_WHEEL; //@@
	int ICC_DECISION_MASK;
	int ICC_DECISION_LIMITED_PERFORMANCE; //@@
	int ICC_DECISION_YAWN;
	int ICC_DECISION_TALKING;
	int ICC_DECISION_DISTRACTION;
	int ICC_DECISION_DROWSINESS;
	int ICC_DECISION_MAKE_PHONE_CALL;
	//int ICC_DECISION_PHONE;
	int ICC_DECISION_SMOKING;
	int ICC_DECISION_AOI_CHANGE; //@@
	int ICC_DECISION_EATING; //@@
	// int ICC_DECISION_DRINKING;
	int ICC_DECISION_IS_DAY_DREAMING; //@@
	int ICC_DECISION_PERSON_ID;
	// int ICC_DECISION_HEAD_GESTURE;
	// int ICC_DECISION_DANGEROUS_BEHAVIOR_DETECT; //@@
	int ICC_DECISION_DRIVER_GENDER_ESTIMATE; 
	int ICC_DECISION_DRIVER_AGE_ESTIMATE; 
	// int ICC_DECISION_SEATBELT; //@@

};


#endif /*__DinsightICC__*/
