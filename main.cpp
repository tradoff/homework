#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int base();
int ROI_AddImg();
int LinearBlending();
int ROI_LinearBlending();
int Split_Merge();
int TrackerBarDemo();
int BrightnessAndContrast();

int BoxFilterDemo();
int GaussianBlurDemo();
int GaussianBlurWithTracker();
int BlurDemo();
int MedianBlurDemo();
int BilateralFilterDemo();

int ErodeDilateDemo();
int MorphologicalGradientDemo();

int CannyDemo();
int SobelDemo();
int LaplacianDemo();
int ScharrDemo();

int ResizeDemo();
int PyrUpDemo();
int PyrDownDemo();

int HoughLinesDemo();
int HoughLinesPDemo();
int HoughCirclesDemo();

int FloodFillDemo();

Mat img;
Mat dstImg;
int threshval = 160;


int g_brightness;
int g_contrast;


Mat g_src = imread("../../../img/002.jpg");
Mat g_dst;
Mat g_logo_src = imread("../../../img/tomato.jpg");
Mat g_logo_dst;

int g_width, g_height, g_sigmaColor, g_sigmaSpace;

int g_type, g_size;

int g_threshold1, g_threshold2;

Mat g_up, g_down;

int g_upDiff, g_loDiff;
Rect g_rectTemp;

int main()
{
	int ret;


	ret = FloodFillDemo();
	
	waitKey();

	return ret;
}

static void onFloodFill(int, void *)
{
	g_dst = g_src.clone();
	floodFill(g_dst, Point(222, 222), Scalar(255, 255, 0), &g_rectTemp,
		Scalar(g_loDiff, g_loDiff, g_loDiff), Scalar(g_upDiff, g_upDiff, g_upDiff), 4);

	imshow("flood fill", g_dst);
}

static void onFloodFillMouseEvent(int event, int x, int y, int, void *)
{
	if (event != CV_EVENT_LBUTTONDOWN)
		return;

	Point seed = Point(x, y);

	g_dst = g_src.clone();
	floodFill(g_dst, seed, Scalar(255, 255, 0), &g_rectTemp,
		Scalar(g_loDiff, g_loDiff, g_loDiff), Scalar(g_upDiff, g_upDiff, g_upDiff), 4);
	imshow("flood fill", g_dst);
}

int FloodFillDemo()
{
	Rect temp;

	g_loDiff = 30;
	g_upDiff = 30;
	namedWindow("flood fill");

	createTrackbar("lower diff", "flood fill", &g_loDiff, 255, onFloodFill);
	createTrackbar("upper diff", "flood fill", &g_upDiff, 255, onFloodFill);
	setMouseCallback("flood fill", onFloodFillMouseEvent);

	onFloodFill(0, 0);

	return 0;
}

int HoughCirclesDemo()
{
	vector<Vec3f> circles;
	Mat gray;

	g_src = imread("../../../img/circles.jpg");
	cvtColor(g_src, gray, CV_BGR2GRAY);
	GaussianBlur(gray, gray, Size(9, 9), 2, 2);
	HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1.5, 10, 100, 100, 0, 0);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Vec3f c = circles[i];
		Point center = Point(c[0], c[1]);
		double radius = c[2];

		circle(g_src, center, 2, Scalar(0, 0, 255), -1);	// center point
		circle(g_src, center, radius, Scalar(0, 0, 255), 2);
		
	}
	imshow("hough circles", g_src);

	return 0;
}

int HoughLinesDemo()
{
	Mat gray;
	vector<Vec2d> lines;

	Canny(g_logo_src, gray, 50, 200);
	cvtColor(gray, g_logo_dst, CV_GRAY2BGR);

	HoughLines(gray, lines, 1, CV_PI / 180, 150);

	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point p1, p2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		double radius = 1000;

		p1.x = cvRound(x0 + radius*(-b));
		p1.y = cvRound(y0 + radius*a);
		p1.x = cvRound(x0 - radius*(-b));
		p1.y = cvRound(y0 - radius*a);

		line(g_logo_dst, p1, p2, Scalar(55, 100, 195));
	}

	imshow("dst", g_logo_dst);

	return 0;
}

int HoughLinesPDemo()
{
	Mat gray;
	vector<Vec4i> lines;

	Canny(g_logo_src, gray, 50, 200);
	cvtColor(gray, g_logo_dst, CV_GRAY2BGR);

	HoughLinesP(gray, lines, 1, CV_PI / 180, 150);

	for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(g_logo_dst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255));
	}

	imshow("dst", g_logo_dst);

	return 0;
}

int PyrDownDemo()
{

	pyrDown(g_logo_src, g_logo_dst);
	imshow("pyr down", g_logo_dst);	// default *0.5

	return 0;
}

int PyrUpDemo()
{

	pyrUp(g_logo_src, g_logo_dst);
	imshow("pyr up", g_logo_dst);	// default *2

	return 0;
}

int ResizeDemo()
{
	resize(g_logo_src, g_up, Size(), 2, 2, INTER_LINEAR);
	resize(g_logo_src, g_down, Size(), 0.5, 0.5, INTER_AREA);

	imshow("up", g_up);
	imshow("down", g_down);
	imshow("src", g_logo_src);

	return 0;
}

int ScharrDemo()
{
	Mat x, y, absX, absY;

	Scharr(g_logo_src, x, -1, 1, 0);
	Scharr(g_logo_src, y, -1, 0, 1);

	convertScaleAbs(x, absX);
	convertScaleAbs(y, absY);

	addWeighted(absX, 0.5, absY, 0.5, 0, g_logo_dst);

	imshow("scharr x", x);
	imshow("scharr y", y);
	imshow("scharr", g_logo_dst);

	return 0;
}

int LaplacianDemo()
{
	Mat gray, edge;

	GaussianBlur(g_logo_src, g_logo_dst, Size(3, 3), 0);
	cvtColor(g_logo_dst, gray, CV_RGB2GRAY);

	Laplacian(gray, edge, -1);

	g_logo_dst = Scalar::all(0);
	g_logo_src.copyTo(g_logo_dst, edge);

	imshow("laplacian", g_logo_dst);

	return 0;
}

static void onSobel(int, void *)
{
	Mat xImg, yImg;
	Mat absXImg, absYImg;

	Sobel(g_logo_src, xImg, -1, 1, 0, g_size*2+1);
	Sobel(g_logo_src, yImg, -1, 0, 1, g_size * 2 + 1);

	convertScaleAbs(xImg, absXImg);
	convertScaleAbs(yImg, absYImg);
	addWeighted(absXImg, 0.5, absYImg, 0.5, 0, g_logo_dst);

	imshow("sobel", g_logo_dst);
}

int SobelDemo()
{

	namedWindow("sobel");

	createTrackbar("kernel size", "sobel", &g_size, 10, onSobel);

	g_size = 3;
	onSobel(0, 0);

	return 0;
}

static void onCanny(int, void *)
{
	Mat edge;
	
	Canny(g_logo_src, edge, g_threshold1, g_threshold2);
	g_logo_dst = Scalar::all(0);
	g_logo_src.copyTo(g_logo_dst, edge);

	imshow("canny", g_logo_dst);
}

int CannyDemo()
{
	g_threshold1 = 3;
	g_threshold1 = 8;
	
	namedWindow("canny");
	createTrackbar("th1", "canny", &g_threshold1, 10, onCanny);
	createTrackbar("th2", "canny", &g_threshold2, 30, onCanny);

	onCanny(0, 0);

	return 0;
}

static void onMorphologicalGradient(int, void *)
{
	Mat kernel = getStructuringElement(MORPH_RECT, Size(g_size * 2 + 1, g_size * 2 + 1));

	morphologyEx(g_src, g_dst, g_type, kernel);

	imshow("morphological gradient", g_dst);
}

int MorphologicalGradientDemo()
{
	g_size = 3;
	g_type = 0;

	namedWindow("morphological gradient");
	createTrackbar("type", "morphological gradient", &g_type, 6, onMorphologicalGradient);
	createTrackbar("size", "morphological gradient", &g_size, 20, onMorphologicalGradient);

	onMorphologicalGradient(0, 0);

	return 0;
}

static void onErodeDilate(int, void *)
{
	Mat element = getStructuringElement(MORPH_RECT, Size(g_size * 2 + 1, g_size * 2 + 1));

	if (g_type)
		erode(g_logo_src, g_logo_dst, element);
	else
		dilate(g_logo_src, g_logo_dst, element);

	imshow("erode dilate", g_logo_dst);
}

int ErodeDilateDemo()
{
	g_type = 0;
	g_size = 5;

	namedWindow("erode dilate");

	createTrackbar("isErode", "erode dilate", &g_type, 1, onErodeDilate);
	createTrackbar("size", "erode dilate", &g_size, 20, onErodeDilate);

	onErodeDilate(0,0);

	return 0;
}

static void onBilateralFilter(int, void *)
{
	bilateralFilter(g_src, g_dst, -1, g_sigmaColor, g_sigmaSpace);

	namedWindow("dst");
	imshow("dst", g_src);
}

int BilateralFilterDemo()
{
	g_sigmaColor = 20;
	g_sigmaSpace = 10;

	namedWindow("dst");

	createTrackbar("sigmaColor", "dst", &g_sigmaColor, 100, onBilateralFilter);
	createTrackbar("sigmaSpace", "dst", &g_sigmaSpace, 100, onBilateralFilter);

	onBilateralFilter(0, 0);


	while (char(waitKey(1)) != 'q');

	return 0;
}

int MedianBlurDemo()
{
	medianBlur(g_src, g_dst, 3);

	namedWindow("src");
	imshow("src", g_src);
	namedWindow("dst");
	imshow("dst", g_dst);

	return 0;
}

static void onGaussianBlur(int, void *)
{
	if (g_width % 2 == 0 || g_height % 2 == 0)
		return;

	GaussianBlur(g_src, g_dst, Size(g_width, g_height), 0, 0);
	imshow("dst", g_dst);
}

int GaussianBlurWithTracker()
{
	g_width = 3;
	g_height = 3;

	namedWindow("dst");

	createTrackbar("width", "dst", &g_width, 20, onGaussianBlur);
	createTrackbar("height", "dst", &g_height, 20, onGaussianBlur);

	onGaussianBlur(0, 0);

	while (char(waitKey(1)) != 'q');
	
	return 0;
}

int BlurDemo()
{
	blur(g_src, g_dst, Size(5, 5));

	namedWindow("src");
	imshow("src", g_src);
	namedWindow("dst");
	imshow("dst", g_dst);

	return 0;
}

int GaussianBlurDemo()
{
	Mat src = imread("../../../img/tomato.jpg");
	Mat dst;

	GaussianBlur(src, dst, Size(5, 5), 1);

	namedWindow("src");
	imshow("src", src);
	namedWindow("dst");
	imshow("dst", dst);

	return 0;
}

int BoxFilterDemo()
{
	Mat src = imread("../../../img/tomato.jpg");
	Mat dst;

	boxFilter(src, dst, -1, Size(5, 5));

	namedWindow("src");
	imshow("src", src);
	namedWindow("dst");
	imshow("dst", dst);

	return 0;
}

void onBrightnessAndContrastChanged(int, void*)
{
	int x, y, c;

	for (y = 0; y < img.rows; y++)
	{
		for (x = 0; x < img.cols; x++)
		{
			for (c = 0; c < 3; c++)
			{
				dstImg.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_contrast*0.01)*(img.at<Vec3b>(y, x)[c]) + g_brightness);
			}
		}
	}
	imshow("brightness and contrast", dstImg);
}

int BrightnessAndContrast() 
{
	img = imread("../../../img/001.jpg");
	dstImg = imread("../../../img/001.jpg");

	g_brightness = 100;
	g_contrast = 50;

	namedWindow("brightness and contrast");
	imshow("brightness and contrast", img);

	createTrackbar("brightness", "brightness and contrast", &g_brightness, 300, onBrightnessAndContrastChanged);
	createTrackbar("contrast", "brightness and contrast", &g_contrast, 200, onBrightnessAndContrastChanged);

	onBrightnessAndContrastChanged(g_brightness, 0);
	onBrightnessAndContrastChanged(g_contrast, 0);

	while (char(waitKey(1)) != 'q');

	return 0;
}

static void on_trackbar(int, void*)  
{  
    Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);  
  
    //定义点和向量  
    vector<vector<Point> > contours;  
    vector<Vec4i> hierarchy;  
  
    //查找轮廓  
    findContours( bw, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );  
    //初始化dst  
    Mat dst = Mat::zeros(img.size(), CV_8UC3);  
    //开始处理  
    if( !contours.empty() && !hierarchy.empty() )  
    {  
        //遍历所有顶层轮廓，随机生成颜色值绘制给各连接组成部分  
        int idx = 0;  
        for( ; idx >= 0; idx = hierarchy[idx][0] )  
        {  
            Scalar color( (rand()&255), (rand()&255), (rand()&255) );  
            //绘制填充轮廓  
            drawContours( dst, contours, idx, color, CV_FILLED, 8, hierarchy );  
        }  
    }  
    //显示窗口  
    imshow( "Connected Components", dst );  
}  

int TrackerBarDemo()
{
	system("color 5F");

    img = imread("../../../img/1.jpg", 0);
    if (!img.data) { printf("Oh，no，读取img图片文件错误~！ \n"); return -1; }

    namedWindow("Image", 1);
    imshow("Image", img);

    namedWindow("Connected Components", 1);

    createTrackbar("Threshold", "Connected Components", &threshval, 255, on_trackbar);
	on_trackbar(threshval, 0);

    waitKey(0);
    return 0;
}

int Split_Merge() {
	Mat src;
	Mat logo = imread("../../../img/tomato.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	vector<Mat> channels;

	src = imread("../../../img/1.jpg");
	split(src, channels);

	Mat imgBlueChannel = channels.at(0);
	addWeighted(imgBlueChannel(Rect(0, 0, logo.cols, logo.rows)), 1, logo, 1, 0, imgBlueChannel(Rect(0, 0, logo.cols, logo.rows)));
	merge(channels, src);
	namedWindow("output0");
	imshow("output0", src);

	src = imread("../../../img/1.jpg");
	Mat imgGreenChannel = channels.at(1);
	addWeighted(imgGreenChannel(Rect(0, 0, logo.cols, logo.rows)), 1, logo, 1, 0, imgGreenChannel(Rect(0, 0, logo.cols, logo.rows)));
	merge(channels, src);
	namedWindow("output1");
	imshow("output1", src);

	src = imread("../../../img/1.jpg");
	Mat imgRedChannel = channels.at(2);
	addWeighted(imgRedChannel(Rect(0, 0, logo.cols, logo.rows)), 1, logo, 1, 0, imgRedChannel(Rect(0, 0, logo.cols, logo.rows)));
	merge(channels, src);
	namedWindow("output2");
	imshow("output2", src);


	return 0;
}

int ROI_LinearBlending()
{
	Mat bg = imread("../../../img/001.jpg");
	Mat logo = imread("../../../img/tomato.jpg");

	Mat imgROI = bg(Rect(0, 0, logo.cols, logo.rows));

	addWeighted(imgROI, 0.5, logo, 0.4, 0, imgROI);

	namedWindow("output");
	imshow("output", bg);

	return 0;
}

int LinearBlending()
{
	Mat src1 = imread("../../../img/002.jpg");
	Mat src2 = imread("../../../img/003.jpg");
	Mat dst;

	addWeighted(src1, 0.5, src2, 0.3, 0, dst);

	namedWindow("output");
	imshow("output", dst);

	return 0;
}

int ROI_AddImg()
{
	Mat bg = imread("../../../img/001.jpg");
	Mat logo = imread("../../../img/tomato.jpg");

	Mat imgROI = bg(Rect(0, 0, logo.cols, logo.rows));

	Mat mask = imread("../../../img/tomato.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	logo.copyTo(imgROI, mask);

	namedWindow("output");
	imshow("output", bg);

	return 0;
}

int base()
{
	Mat img = imread("../../../img/1.jpg", CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
	namedWindow("best img");
	imshow("best img", img);

	img = imread("../../../img/1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	namedWindow("gray img");
	imshow("gray img", img);

	img = imread("../../../img/1.jpg", CV_LOAD_IMAGE_COLOR);
	namedWindow("default img");
	imshow("default img", img);

	vector<int> compress_params;
	compress_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compress_params.push_back(9);

	try
	{
		imwrite("../../../img/2.png", img, compress_params);
	}
	catch (runtime_error &err)
	{
		cout << "imwrite failed..." << err.what() << endl;
	}

	return 0;
}