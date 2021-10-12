// FairMOT.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include<chrono>
#include "tracker.h"

void draw_tracks(cv::Mat&img, std::string cls, std::vector<std::shared_ptr<STrack>>& tracks)
{
	for (auto& tr:tracks)
	{
		cv::Rect_<float> loc = tr->TlwhToRect();
		int color_idx = (tr->GetTrackID()) % kColorNum * 4;
		cv::Scalar_<int> color = cv::Scalar_<int>(kColorArray[color_idx], kColorArray[color_idx + 1], kColorArray[color_idx + 2], kColorArray[color_idx + 3]);
		cv::rectangle(img, loc, color, 3, cv::LINE_AA, 0);

		int fontCalibration = cv::FONT_HERSHEY_COMPLEX;
		float fontScale = 0.6; //0.6; //1.2f;
		int fontThickness = 1; //1; // 2;
		char text[15];
		//sprintf(text, "%d:%.2f", tr->track_id, traj.back().detection_score);
		sprintf(text, "%d", tr->GetTrackID());
		std::string buff = text;
		putText(img, cls+":"+buff, cv::Point(loc.x, loc.y), fontCalibration, fontScale, color, fontThickness);/**/
	}
}

std::string get_cls_index(int cls) {
	std::string str_cls = "";
	if (cls == 0) {
		str_cls = "car";
	}
	if (cls == 1) {
		str_cls = "bicycle";
	}
	if (cls == 2) {
		str_cls = "person";
	}
	if (cls == 3) {
		str_cls = "cyclist";
	}
	if (cls == 4) {
		str_cls = "tricycle";
	}
	return str_cls;
}

void main()
{
	const int num_class = 5;
	DetectorConfig detconfig;
	detconfig.method = DetectorMethod::FromFairMOT;
	detconfig.fairmot.threshold = 0.6f;
	detconfig.fairmot.ltrb = true;
	detconfig.fairmot.model_file = "../models/mcmot_last_track_hrnet_18_deconv";// model_54_crowdhuman_1088x608     mix_ft_ch_dlav0_15
	Detection* det = DetectorFactory::create_object(detconfig);
	if(!det->InitDetection()) 
		return; 

	JDETrackerConfig config;
	config.conf_thres = 0.4f;
	config.track_buffer = 30;
	config.num_class = 5;
	config.max_lost_time = 30;
	int frame_rate = 30;

	JDETracker jde(config, frame_rate);
	cv::VideoCapture capture;
	capture.open("../video/MOT16-03.mp4");  //"video/MOT16-03.mp4"
	cv::Mat frame;
	std::map<int, std::vector<DetectionBox>> vec_db;
	std::map<int, std::vector<cv::Mat>> vec_features;
	int frame_index = 0;
	std::string window_name = "tracking";
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);

	cv::VideoWriter filesavewriter;
	while (true)
	{
 		capture >> frame;
		if (frame.empty()) break;
		 if (frame_index == 0)
 		{
 			filesavewriter.open("../video/result.mp4", cv::VideoWriter::fourcc('X', 'V', 'I', 'D')
 				, frame_rate, cv::Size(frame.cols, frame.rows), true);
 		}
		auto detect_start = std::chrono::steady_clock::now();
		det->GetDetection(frame, num_class, vec_db, vec_features);
		auto detect_end = std::chrono::steady_clock::now();
		auto detection_time = std::chrono::duration_cast<std::chrono::milliseconds>(detect_end - detect_start).count();

		auto track_start = std::chrono::steady_clock::now();
		std::map < int, std::vector<std::shared_ptr<STrack>>> tracks = jde.UpdateTracking(vec_db, vec_features);

		auto track_end = std::chrono::steady_clock::now(); 
		auto tracking_time = std::chrono::duration_cast<std::chrono::milliseconds>(track_end - track_start).count();
		float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(track_end - detect_start).count();
		std::cout << "detection:" << detection_time << ", tracking:" << tracking_time << ", fps:" << inference_fps << std::endl;
		for (int cls = 0; cls < num_class; cls++) {
			std::string cls_str = get_cls_index(cls);
			draw_tracks(frame, cls_str, tracks[cls]);
		}
		filesavewriter.write(frame);
		cv::imshow(window_name, frame);
		char c = (char)cv::waitKey(1);
		if (c == 27)
			break;
		frame_index++;

	}
	filesavewriter.release();

}
