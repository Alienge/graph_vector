#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/core/core.hpp"

#include "opencv2/imgproc/imgproc.hpp"
#include<iostream>

using namespace cv;
void main()
{
//求出图像的梯度方向
	Mat img,img_gray;
	img = imread("D://chicago.jpg");
	std::cout << img.type() << std::endl;
	int ddepth = CV_16S;
	cvtColor(img, img_gray, CV_RGB2GRAY);
	Mat Z(img_gray.size(), CV_16S, Scalar(0));
	//std::cout << Z << std::endl;
	Mat grad_x, grad_y;
	Mat grad_x_uint, grad_y_uint;
	double scale = 1;
	double delta = 0;
	double minv_x = 0.0, maxv_x = 0.0;
	double* minp_x = &minv_x;
	double* maxp_x = &maxv_x;

	double minv_y = 0.0, maxv_y = 0.0;
	double* minp_y = &minv_y;
	double* maxp_y = &maxv_y;

	//求x方向上的梯度
	Sobel(img_gray,grad_x,ddepth,1,0,3,scale,delta,BORDER_DEFAULT);
	//求y方向上的梯度
	Sobel(img_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

	//准备压缩梯度到0-255（x和y方向上）
	minMaxIdx(grad_x, minp_x, maxp_x);
	grad_x_uint = (grad_x - minv_x)* 255.0 / (maxv_x - minv_x);

	minMaxIdx(grad_y, minp_y, maxp_y);
	grad_y_uint = (grad_y - minv_y)* 255.0 / (maxv_y - minv_y);

    //为显示图片，合并x，y梯度通道再加上一个常数通道
	std::vector<Mat> channel_merge;
	channel_merge.push_back(grad_x_uint);
	channel_merge.push_back(grad_y_uint);
	channel_merge.push_back(Z);
	Mat vec_field_temp;
	merge(channel_merge, vec_field_temp);
	//std::cout << grad_y.size() << std::endl;
	//std::cout << maxv << std::endl;
	//std::cout << grad_x_uint.type() << std::endl;
	//std::cout << grad_x_uint  << std::endl;
	Mat vec_field;
	vec_field_temp.convertTo(vec_field, CV_8UC3);
	std::cout << vec_field.type() << std::endl;
	imshow("dst", vec_field);
	std::cout << "test end" << std::endl;
	waitKey(0);

}