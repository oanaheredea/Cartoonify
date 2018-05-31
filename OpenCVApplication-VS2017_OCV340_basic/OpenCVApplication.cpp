// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

Mat testCanny(Mat src)
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);

		return dst;
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}
/*******************************************************************************************************************************/

Mat OpenImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, 0);
		return src;
	}
}

Mat OpenColorImage()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src;
		src = imread(fname, 1);
		return src;
	}
}

// quantize the image to numBits 
Mat quantizeImage(Mat src, int numBits)
{
	Mat dst = src.clone();

	uchar maskBit = 0xFF;

	maskBit = maskBit << (8 - numBits);

	for (int j = 0; j < dst.rows; j++)
		for (int i = 0; i < dst.cols; i++)
		{
			cv::Vec3b valVec = dst.at<cv::Vec3b>(j, i);
			valVec[0] = valVec[0] & maskBit;
			valVec[1] = valVec[1] & maskBit;
			valVec[2] = valVec[2] & maskBit;
			dst.at<cv::Vec3b>(j, i) = valVec;
		}

	return dst;
}

void quantizeImage2(Mat *src, int numBits)
{
	//Mat dst = (*src).clone();

	uchar maskBit = 0xFF;

	maskBit = maskBit << (8 - numBits);

	for (int j = 0; j < (*src).rows; j++)
		for (int i = 0; i < (*src).cols; i++)
		{
			cv::Vec3b valVec = (*src).at<cv::Vec3b>(j, i);
			valVec[0] = valVec[0] & maskBit;
			valVec[1] = valVec[1] & maskBit;
			valVec[2] = valVec[2] & maskBit;
			(*src).at<cv::Vec3b>(j, i) = valVec;
		}
}

void quantization()
{
	Mat in = OpenColorImage();
	Mat quantizedImage = quantizeImage(in, 3);		
	imshow("Q", quantizedImage);
	waitKey(0);
}

int compareMyType(const void * a, const void * b)
{
	if (*(float*)a <  *(float*)b) return -1;
	if (*(float*)a == *(float*)b) return 0;
	if (*(float*)a >  *(float*)b) return 1;
}

void medianFilter(Mat src, Mat dst, int kernelSize)
{
	int halfKernelSize = kernelSize / 2;
	float *neighbourhood = (float*)calloc(sizeof(float), (kernelSize * kernelSize));

	for (int i = 0 + halfKernelSize; i<(src.rows - halfKernelSize); i++)
		for (int j = 0 + halfKernelSize; j<(src.cols - halfKernelSize); j++)
		{
			for (int ii = -halfKernelSize; ii<halfKernelSize + 1; ii++)
				for (int jj = -halfKernelSize; jj<halfKernelSize + 1; jj++)
					neighbourhood[(ii + halfKernelSize)*kernelSize + (jj + halfKernelSize)] = src.data[(i + ii) * src.cols + (j + jj)];
			qsort(neighbourhood, kernelSize*kernelSize, sizeof(float), compareMyType);
			dst.data[(i)* src.cols + (j)] = neighbourhood[(kernelSize*kernelSize) / 2 + 1];
		}
	double t = (double)getTickCount(); 
	t = ((double)getTickCount() - t) / getTickFrequency();

}

void medianCall()
{
	Mat src = OpenImage();
	Mat dst(src.rows, src.cols, CV_8UC1);
	int kernelSize;
	printf("Enter the kernel size: \n");
	scanf("%d", &kernelSize);
	medianFilter(src, dst, kernelSize);
	imshow("SRC", src);
	imshow("DST", dst);
	waitKey(0);
}

Mat convolution(Mat src, int kernel[9])
{
	Mat dst(src.rows, src.cols, CV_8UC1);
	int di[9] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
	int dj[9] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int sum = 0;
	int maxim = 0;
	int minim = INT_MAX;
	int max_abs = 0;
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			sum = 0;
			for (int k = 0; k < 9; k++)
			{
				sum += kernel[k] * src.at<uchar>(i + di[k], j + dj[k]);
			}
			if (sum > maxim)
				maxim = sum;
			if (sum <= minim)
				minim = sum;
		}
	}
	if (minim < 0)
		minim *= (-1);

	max_abs = max(maxim, minim);
	printf("MAX %d \n", max_abs);
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			sum = 0;
			for (int k = 0; k < 9; k++)
			{
				sum += kernel[k] * src.at<uchar>(i + di[k], j + dj[k]);
			}
			if (sum < 0)
				dst.data[i * src.cols + j] = (-1) * sum * 255 / max_abs;
			else
				dst.data[i * src.cols + j] = sum * 255 / max_abs;
		}
	}
	return dst;
}

void computeSobelXY()
{
	Mat src = OpenImage();
	Mat dst(src.rows, src.cols, CV_8UC1);
	Mat dst2(src.rows, src.cols, CV_8UC1);

	int sobelY[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	int sobelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	dst = convolution(src, sobelX);
	dst2 = convolution(src, sobelY);
	waitKey(0);
}

Mat computeMagnitude(Mat src)
{
	Mat rez(src.rows, src.cols, CV_8UC1);
	Mat dst(src.rows, src.cols, CV_8UC1);
	Mat dst2(src.rows, src.cols, CV_8UC1);
	int sobelY[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	int sobelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	dst = convolution(src, sobelX);
	dst2 = convolution(src, sobelY);
	int max = 0;
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			rez.at<uchar>(i, j) = sqrt(dst.at<uchar>(i, j) * dst.at<uchar>(i, j) + dst2.at<uchar>(i, j) * dst2.at<uchar>(i, j));
			if (rez.at<uchar>(i, j) > max)
				max = rez.at<uchar>(i, j);
		}
	}
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			rez.at<uchar>(i, j) = rez.at<uchar>(i, j) * 255 / max;
		}
	}
	return rez;
}

Mat computeDirection(Mat src)
{
	Mat rez(src.rows, src.cols, CV_8UC1);
	Mat dst(src.rows, src.cols, CV_8UC1);
	Mat dst2(src.rows, src.cols, CV_8UC1);
	int sobelY[9] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };
	int sobelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	dst = convolution(src, sobelX);
	dst2 = convolution(src, sobelY);
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			rez.at<uchar>(i, j) = atan2(dst2.at<uchar>(i, j), dst.at<uchar>(i, j)) * 180 / PI;
		}
	}
	return rez;
}

Mat nonMaximaSuppression(Mat src)
{
	Mat rez(src.rows, src.cols, CV_8UC1);
	Mat mag(src.rows, src.cols, CV_8UC1);
	Mat img(src.rows, src.cols, CV_8UC1);

	float deg = 22.5;
	int dir;
	mag = computeMagnitude(src);
	mag.copyTo(img);
	rez = computeDirection(src);
	for (int i = 1; i < src.rows - 1; i++)
	{
		for (int j = 1; j < src.cols - 1; j++)
		{
			if ((rez.at<uchar>(i, j) >= 0 && rez.at<uchar>(i, j) <= deg) || (rez.at<uchar>(i, j) >(315 + deg) && rez.at<uchar>(i, j) <= (315 + deg + deg)) || (rez.at<uchar>(i, j) >(180 - deg) && rez.at<uchar>(i, j) <= (180 + deg)))
			{
				dir = 2;
				if (mag.at<uchar>(i, j) < max(mag.at<uchar>(i, j + 1), mag.at<uchar>(i, j - 1)))
					img.at<uchar>(i, j) = 0;
				else
					img.at<uchar>(i, j) = mag.at<uchar>(i, j);
			}
			else if ((rez.at<uchar>(i, j) > (45 - deg) && rez.at<uchar>(i, j) <= (45 + deg)) || (rez.at<uchar>(i, j) > (225 - deg) && rez.at<uchar>(i, j) <= (225 + deg)))
			{
				dir = 1;
				if (mag.at<uchar>(i, j) < max(mag.at<uchar>(i + 1, j + 1), mag.at<uchar>(i - 1, j - 1)))
					img.at<uchar>(i, j) = 0;
				else
					img.at<uchar>(i, j) = mag.at<uchar>(i, j);
			}
			else if ((rez.at<uchar>(i, j) > (90 - deg) && rez.at<uchar>(i, j) <= (90 + deg)) || (rez.at<uchar>(i, j) > (270 - deg) && rez.at<uchar>(i, j) <= (270 + deg)))
			{
				dir = 0;
				if (mag.at<uchar>(i, j) < max(mag.at<uchar>(i - 1, j), mag.at<uchar>(i + 1, j)))
					img.at<uchar>(i, j) = 0;
				else
					img.at<uchar>(i, j) = mag.at<uchar>(i, j);
			}
			else if ((rez.at<uchar>(i, j) > (135 - deg) && rez.at<uchar>(i, j) <= (135 + deg)) || (rez.at<uchar>(i, j) > (315 - deg) && rez.at<uchar>(i, j) <= (315 + deg)))
			{
				dir = 3;
				if (mag.at<uchar>(i, j) < max(mag.at<uchar>(i - 1, j - 1), mag.at<uchar>(i + 1, j + 1)))
					img.at<uchar>(i, j) = 0;
				else
					img.at<uchar>(i, j) = mag.at<uchar>(i, j);
			}
		}
	}
	return img;
}

Mat convert_RGB_grayscale(Mat img)
{
	int row = img.rows;
	int col = img.cols;
	Mat gray(row, col, CV_8UC1);
	uchar r = 0, g = 0, b = 0;

	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			r = img.data[3 * (i * img.cols + j) + 2];
			g = img.data[3 * (i * img.cols + j) + 1];
			b = img.data[3 * (i * img.cols + j)];

			gray.at<uchar>(i, j) = (r + g + b) / 3;
		}
	}

	return gray;
}

int* computeHistogram(Mat img) {
	int *hist = (int*)malloc(sizeof(int) * 256);

	for (int i = 0; i < 256; i++) {
		hist[i] = 0;
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			hist[img.at<uchar>(i, j)]++;
		}
	}

	return hist;
}

Mat tresholding(Mat img)
{
	Mat dst(img.rows, img.cols, CV_8UC1);
	int *hist = computeHistogram(img);
	int imax = INT_MIN;
	int imin = INT_MAX;
	int T = 0;
	int Tk = 0;
	int ug1 = 0;
	int ug2 = 0;
	int n1 = 0;
	float error = 0.1;
	int n2 = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			int pixel = img.data[i * img.cols + j];
			if (pixel > imax)
				imax = pixel;
			if (pixel < imin)
				imin = pixel;
		}
	}
	T = (imax + imin) / 2;
	while (T - Tk < error)
	{
		Tk = T;
		for (int g = imin; g < T; g++)
		{
			ug1 += g * hist[g];
		}
		for (int g = imin; g < T; g++)
		{
			n1 += hist[g];
		}

		if (n1 != 0)
			ug1 /= n1;

		for (int g = T + 1; g < imax; g++)
		{
			ug2 += g * hist[g];
		}
		for (int g = T + 1; g < imax; g++)
		{
			n2 += hist[g];
		}

		if (n2 != 0)
			ug2 /= n2;

		T = (ug1 + ug2) / 2;
	}
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			if (img.data[i*img.cols + j] < T)
				dst.data[i*img.cols + j] = 0;
			else
				dst.data[i* img.cols + j] = 255;
		}
	}
	return dst;
}

Mat dilation(Mat src, int n)
{
	Mat dst;
	Mat aux;
	src.copyTo(aux);
	src.copyTo(dst);
	int height = src.rows;
	int width = src.cols;

	for (int k = 0; k < n; k++)
	{
		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				uchar pixel = aux.at<uchar>(i, j);

				if (pixel < 255)
				{
					dst.at<uchar>(i, j + 1) = 0;
					dst.at<uchar>(i + 1, j) = 0;
				}
			}
		}
		dst.copyTo(aux);
	}
	return dst;
}

void mouseEvent(int evt, int x, int y, int flags, void* param)
{
	Mat* rgb = (Mat*)param;
	if (evt == CV_EVENT_LBUTTONDOWN)
	{
		quantizeImage2(rgb, 3);
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*rgb).at<Vec3b>(y, x)[2],
			(int)(*rgb).at<Vec3b>(y, x)[1],
			(int)(*rgb).at<Vec3b>(y, x)[0]);
	}
}


void callGradient()
{
	Mat in = OpenColorImage();
	//Mat in2 = in.clone();
	Mat q = quantizeImage(in, 3);
	Mat bw(in.rows, in.cols, CV_8UC1);
	bw = convert_RGB_grayscale(q);
	Mat contour(in.rows, in.cols, CV_8UC1);
	/*
	namedWindow("My Window", 1);
	setMouseCallback("My Window", mouseEvent, &in2);
	imshow("My Window", in2);
	*/
	
	for (int i = 0; i < contour.rows; i++)
	{
		for (int j = 0; j < contour.cols; j++)
		{
			contour.data[i * contour.cols + j] = 255;
		}
	}	
	int gaussian[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	convolution(bw, gaussian);
	Mat dst = nonMaximaSuppression(bw);
	for (int i = 0; i < q.rows; i++)
	{
		for (int j = 0; j < q.cols; j++)
		{
			if (dst.data[i * dst.cols + j] < 128)
			{
				dst.data[i * dst.cols + j] = 255;
			}
			else
				dst.data[i * dst.cols + j] = 0;
		}
	}
	
	Mat r = dilation(dst, 1);
	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if (r.data[i * r.cols + j] == 0)
			{
				q.data[3 * (i* q.cols + j) + 2] = 0;
				q.data[3 * (i* q.cols + j) + 1] = 0;
				q.data[3 * (i* q.cols + j) + 0] = 0;
			}
		}
	}

	imshow("Q", q);
	imshow("SRC", in);
	
	waitKey(0);
}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Quantize Image\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				//testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				callGradient();
				break;
		}
	}
	while (op!=0);
	return 0;
}