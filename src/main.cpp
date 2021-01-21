#include <iostream>
#include <vector>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include "paddle_api.h"             // NOLINT
#include "paddle_use_kernels.h"     // NOLINT
#include "paddle_use_ops.h"         // NOLINT
#include "TMtCNN.h"
#include "TArcface.h"
#include "TRetina.h"
#include "TWarp.h"
#include "TLive.h"
#include "TBlur.h"
//----------------------------------------------------------------------------------------
//
// Created by markson zhang
//
// Edited by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------
// Build defines
// comment them to turn a function off
//----------------------------------------------------------------------------------------
#define RETINA                  //comment if you want to use MtCNN landmark detection instead
#define RECOGNIZE_FACE
#define TEST_LIVING
#define TEST_MASK
#define AUTO_FILL_DATABASE
#define BLUR_FILTER_STRANGER
// some diagnostics
#define SHOW_LEGEND
#define SHOW_LANDMARKS
//----------------------------------------------------------------------------------------
// Adjustable Parameters
//----------------------------------------------------------------------------------------
const int   MaxItemsDatabase = 2000;
const int   MinHeightFace    = 90;
const float MinFaceThreshold = 0.50;
const float FaceLiving       = 0.93;
const double MaxBlur         = -25.0;   //more positive = sharper image
const double MaxAngle        = 10.0;
//----------------------------------------------------------------------------------------
// Some globals
//----------------------------------------------------------------------------------------
const int   RetinaWidth      = 320;
const int   RetinaHeight     = 240;
float ScaleX, ScaleY;
vector<cv::String> NameFaces;
bool FlagMask;
//----------------------------------------------------------------------------------------
using namespace std;
using namespace cv;
using namespace paddle::lite_api;  // NOLINT

//----------------------------------------------------------------------------------------
//  Computing the cosine distance between input feature and ground truth feature
//----------------------------------------------------------------------------------------
inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2)
{
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}
//----------------------------------------------------------------------------------------
// painting
//----------------------------------------------------------------------------------------
void DrawObjects(cv::Mat &frame, vector<FaceObject> &Faces)
{
    for(size_t i=0; i < Faces.size(); i++){
        FaceObject& obj = Faces[i];

//----- rectangle around the face -------
        obj.rect.x *= ScaleX;
        obj.rect.y *= ScaleY;
        obj.rect.width *= ScaleX;
        obj.rect.height*= ScaleY;
        cv::rectangle(frame, obj.rect, cv::Scalar(0, 255, 0));
//---------------------------------------

//----- diagnostic ----------------------
#ifdef SHOW_LANDMARKS
        for(int u=0;u<5;u++){
            obj.landmark[u].x*=ScaleX;
            obj.landmark[u].y*=ScaleY;
        }

        cv::circle(frame, obj.landmark[0], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[1], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[2], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[3], 2, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, obj.landmark[4], 2, cv::Scalar(0, 255, 255), -1);
#endif // SHOW_LANDMARKS
//---------------------------------------
#ifdef SHOW_LEGEND
        cv::putText(frame, cv::format("Angle : %0.1f", obj.Angle),cv::Point(10,40),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        cv::putText(frame, cv::format("Face prob : %0.4f", obj.FaceProb),cv::Point(10,60),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        cv::putText(frame, cv::format("Name prob : %0.4f", obj.NameProb),cv::Point(10,80),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
#ifdef TEST_LIVING
        if(obj.Color==2 || obj.NameIndex==-4){
            //face is too tiny or is wearing a mask
            cv::putText(frame, cv::format("Live prob : ??"),cv::Point(10,100),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        }
        else{
            //face is ok
            cv::putText(frame, cv::format("Live prob : %0.4f", obj.LiveProb),cv::Point(10,100),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        }
#endif // TEST_LIVING
#ifdef TEST_MASK
        if(FlagMask){
            //face with mask
            cv::putText(frame, cv::format("Mask : ON"),cv::Point(10,120),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        }
        else{
            //face without mask
            cv::putText(frame, cv::format("Mask : OFF"),cv::Point(10,120),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));
        }
#endif // TEST_MASK
#endif // SHOW_LEGEND
//----- labels ----------------------------
#ifdef RECOGNIZE_FACE
        cv::String Str;
        cv::Scalar color;
        int  baseLine = 0;

        switch(obj.Color){
            case 0 : color = cv::Scalar(255, 255, 255); break;  //default white -> face ok
            case 1 : color = cv::Scalar( 80, 255, 255); break;  //yellow ->stranger
            case 2 : color = cv::Scalar(255, 237, 178); break;  //blue -> too tiny
            case 3 : color = cv::Scalar(127, 127, 255); break;  //red -> fake
            case 4 : color = cv::Scalar( 32,  32, 255); break;  //pure red -> mask
            default: color = cv::Scalar(255, 255, 255);
        }

        switch(obj.NameIndex){
            case -1: Str="Stranger"; break;
            case -2: Str="too tiny"; break;
            case -3: Str="Fake !";   break;
            case -4: Str="Mask !";   break;
            default: Str=NameFaces[obj.NameIndex];
        }

        cv::Size label_size = cv::getTextSize(Str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if(y<0) y = 0;
        if(x+label_size.width > frame.cols) x=frame.cols-label_size.width;

        cv::rectangle(frame, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),color, -1);
        cv::putText(frame, Str, cv::Point(x, y+label_size.height+2),cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0));
#endif // RECOGNIZE_FACE
    }
}
//----------------------------------------------------------------------------------------
// main
//----------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
    float f;
    float FPS[16];
    int   n,Fcnt=0;
    size_t i;
    cv::Mat frame;
    cv::Mat result_cnn;
    cv::Mat faces;
    std::vector<FaceObject> Faces;
    vector<cv::Mat> fc1;
    string pattern_jpg = "./img/*.jpg";
    cv::String NewItemName;

    size_t FaceCnt;
    int classify_w = 128;
    int classify_h = 128;
    float scale_factor = 1.f / 256;
    int FaceImgSz  = classify_w * classify_h;
    // Mask detection (second phase, when the faces are located)
    MobileConfig Mconfig;
    std::shared_ptr<PaddlePredictor> Mpredictor;

    //the networks
    TLive Live;
    TWarp Warp;
    TMtCNN MtCNN;
    TArcFace ArcFace;
    TRetina Rtn(RetinaWidth, RetinaHeight);
    TBlur Blur;
    //some timing
    chrono::steady_clock::time_point Tbegin, Tend;

    Live.LoadModel();

    for(i=0;i<16;i++) FPS[i]=0.0;

    //load mask detection model
    Mconfig.set_model_from_file("./models/mask/mask_detector_opt2.nb");
    Mpredictor = CreatePaddlePredictor<MobileConfig>(Mconfig);
    std::cout << "Load classification model succeed." << std::endl;

    // Get Input Tensor
    std::unique_ptr<Tensor> input_tensor1(std::move(Mpredictor->GetInput(0)));
    input_tensor1->Resize({1, 3, classify_h, classify_w});

    // Get Output Tensor
    std::unique_ptr<const Tensor> output_tensor1(std::move(Mpredictor->GetOutput(0)));

    //OpenCV Version
    cout << "OpenCV Version: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "."
    << CV_SUBMINOR_VERSION << endl;
    cout << " " << endl;

#ifdef RECOGNIZE_FACE
    cout << "Trying to recognize faces" << endl;
    cout << " " << endl;
#ifdef RETINA
    cout << "Using Retina" << endl;
    cout << " " << endl;
#else
    cout << "Using MtCNN" << endl;
    cout << " " << endl;
#endif // RETINA

#ifdef TEST_MASK
    cout << "Test mask" << endl;
    cout << " " << endl;
#endif // TEST_LIVING

#ifdef TEST_LIVING
    cout << "Test living or fake face" << endl;
    cout << " " << endl;
#endif // TEST_LIVING

#ifdef AUTO_FILL_DATABASE
    cout << "Automatic adding strangers to database" << endl;
    cout << " " << endl;
#ifdef BLUR_FILTER_STRANGER
    cout << "Blur filter - only sharp images to database" << endl;
    cout << " " << endl;
#endif // BLUR_FILTER_STRANGER
#endif // AUTO_FILL_DATABASE
#endif // RECOGNIZE_FACE

    //if you like to load a picture of a face into the database
    //give the name of the .jpg image as argument on the command line
    //without arguments the app will run the .mp4 video or use the camera
    if(argc>1){
        const char* imagepath = argv[1];

        cv::Mat frame = cv::imread(imagepath, 1);
        if(frame.empty()){
            fprintf(stderr, "cv::imread %s failed\n", imagepath);
            return -1;
        }
        //extract
        ScaleX = ((float) frame.cols) / RetinaWidth;
        ScaleY = ((float) frame.rows) / RetinaHeight;
        // copy/resize image to result_cnn as input tensor
        cv::resize(frame, result_cnn, Size(RetinaWidth,RetinaHeight),INTER_LINEAR);
        //get the face
        Rtn.detect_retinaface(result_cnn,Faces);
        //only one face per picture
        if(Faces.size()==1){
            if(Faces[0].FaceProb>MinFaceThreshold){
                //get centre aligned image
                cv::Mat aligned = Warp.Process(result_cnn,Faces[0]);

                cv::String Str = imagepath;
                n   = Str.rfind('/');
                Str = Str.erase(0,n+1);
                Str = Str.erase(Str.length()-4, Str.length()-1);  //remove .jpg


                imwrite("./img/"+Str+".jpg", aligned);
                cout << "Stored to database : " << Str << endl;
            }
        }
        return 0;
    }


    //loading the faces
	cv::glob(pattern_jpg, NameFaces);
    FaceCnt=NameFaces.size();
	if(FaceCnt==0) {
		cout << "No image files[jpg] in database" << endl;
	}
	else{
        cout << "Found "<< FaceCnt << " pictures in database." << endl;
        for(i=0; i<FaceCnt; i++){
            //convert to landmark vector and store into fc
            faces = cv::imread(NameFaces[i]);
            fc1.push_back(ArcFace.GetFeature(faces));
            //get a proper name
            cv::String &Str = NameFaces[i];
            n   = Str.rfind('/');
            Str = Str.erase(0,n+1);
            n   = Str.find('#');
            if(n>0) Str = Str.erase(n,Str.length()-1);                //remove # some numbers.jpg
            else    Str = Str.erase(Str.length()-4, Str.length()-1);  //remove .jpg
            if(FaceCnt>1) printf("\rloading: %.2lf%% ",(i*100.0)/(FaceCnt-1));
        }
        cout << "" << endl;
        cout << "Loaded "<<FaceCnt<<" faces in total"<<endl;
    }

    // RaspiCam or Norton_2.mp4 ?
    // cv::VideoCapture cap(0);             //RaspiCam
    cv::VideoCapture cap("Norton_M.mp4");   //Movie
    if (!cap.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return 0;
    }
    cout << "Start grabbing, press ESC on TLive window to terminate" << endl;

    while(1){
        cap >> frame;
        if (frame.empty()) {
            cerr << "End of movie" << endl;
            break;
        }
        ScaleX = ((float) frame.cols) / RetinaWidth;
        ScaleY = ((float) frame.rows) / RetinaHeight;

        // copy/resize image to result_cnn as input tensor
        cv::resize(frame, result_cnn, Size(RetinaWidth,RetinaHeight),INTER_LINEAR);

        Tbegin = chrono::steady_clock::now();

#ifdef RETINA
        Rtn.detect_retinaface(result_cnn,Faces);
#else
        MtCNN.detect(result_cnn,Faces);
#endif // RETINA

#ifdef RECOGNIZE_FACE
        //reset indicators
        for(i=0;i<Faces.size();i++){
            Faces[i].NameIndex = -2;    //-2 -> too tiny (may be negative to signal the drawing)
            Faces[i].Color     =  2;
            Faces[i].NameProb  = 0.0;
            Faces[i].LiveProb  = 0.0;
        }
        //run through the faces only when you got one face.
        //more faces (if large enough) are not a problem
        //in this app with an input image of 324x240, they become too tiny
        if(Faces.size()==1){
            //looks stupid, running through a loop of size 1
            //however, for your convenience using [i]
            for(i=0;i<Faces.size();i++){
                float x1 = Faces[i].rect.x;
                float y1 = Faces[i].rect.y;
                float x2 = Faces[i].rect.width+x1;
                float y2 = Faces[i].rect.height+y1;
#ifdef TEST_MASK
                //test to see if the face is wearing a mask
                auto* input_data = input_tensor1->mutable_data<float>();
                //enlarge 10%
                float w = Faces[i].rect.width/20.0;
                float h = Faces[i].rect.height/20.0;
                cv::Point pt1(std::max(x1-w,float(0.0)),std::max(y1-h,float(0.0)));
                cv::Point pt2(std::min(x2+w,float(frame.cols)),std::min(y2+h,float(frame.rows)));
                //RecClip is completly inside the frame
                cv::Rect  RecClip(pt1, pt2);
                cv::Mat   resized_img;
                cv::Mat   imgf;

                if(RecClip.width>0 && RecClip.height>0){
                    //roi has size RecClip
                    cv::Mat roi = frame(RecClip);

                    //resized_img has size 128x128 (uchar)
                    cv::resize(roi, resized_img, cv::Size(classify_w, classify_h), 0.f, 0.f, cv::INTER_CUBIC);

                    //imgf has size 128x128 (float in range 0.0 - +1.0)
                    resized_img.convertTo(imgf, CV_32FC3, scale_factor);

                    //input tensor has size 128x128 (float in range -0.5 - +0.5)
                    // fill tensor with mean and scale and trans layout: nhwc -> nchw, neon speed up
                    //offset_nchw(n, c, h, w) = n * CHW + c * HW + h * W + w
                    //offset_nhwc(n, c, h, w) = n * HWC + h * WC + w * C + c
                    const float* dimg = reinterpret_cast<const float*>(imgf.data);

                    float* dout_c0 = input_data;
                    float* dout_c1 = input_data + FaceImgSz;
                    float* dout_c2 = input_data + FaceImgSz * 2;

                    for(int i=0;i<FaceImgSz;i++){
                        *(dout_c0++) = (*(dimg++) - 0.5);
                        *(dout_c1++) = (*(dimg++) - 0.5);
                        *(dout_c2++) = (*(dimg++) - 0.5);
                    }

                    // Classification Model Run
                    Mpredictor->Run();

                    auto* outptr = output_tensor1->data<float>();

                    FlagMask = outptr[1] > 0.5f;
                    if(FlagMask){
                        Faces[i].NameIndex = -4;    //a mask
                        Faces[i].Color     =  4;
                    }
                }
#endif // TEST_MASK
#ifdef TEST_MASK
                //try to recognize only people without a mask
                if(Faces[i].FaceProb>MinFaceThreshold && !FlagMask){
#else
                if(Faces[i].FaceProb>MinFaceThreshold){
#endif // TEST_MASK
                    //get centre aligned image
                    cv::Mat aligned = Warp.Process(result_cnn,Faces[i]);
                    Faces[i].Angle  = Warp.Angle;
                    //features of camera image
                    cv::Mat fc2 = ArcFace.GetFeature(aligned);
                    //reset indicators
                    Faces[i].NameIndex = -1;    //a stranger
                    Faces[i].Color     =  1;
                    //the similarity score
                    if(FaceCnt>0){
                        vector<double> score_;
                        for(size_t c=0;c<FaceCnt;c++) score_.push_back(CosineDistance(fc1[c], fc2));
                        int Pmax = max_element(score_.begin(),score_.end()) - score_.begin();
                        Faces[i].NameIndex = Pmax;
                        Faces[i].NameProb  = score_[Pmax];
                        score_.clear();
                        if(Faces[i].NameProb >= MinFaceThreshold){
                            //recognize a face
                            if(Faces[i].rect.height < MinHeightFace){
                                Faces[i].Color = 2; //found face in database, but too tiny
                            }
                            else{
                                Faces[i].Color = 0; //found face in database and of good size
#ifdef TEST_LIVING
                                //test fake face
                                struct LiveFaceBox LiveBox={x1,y1,x2,y2};

                                Faces[i].LiveProb=Live.Detect(result_cnn,LiveBox);
                                if(Faces[i].LiveProb<=FaceLiving){
                                    Faces[i].Color     =  3; //fake
                                    Faces[i].NameIndex = -3;
                                }
#endif // TEST_LIVING
                            }
                        }
                        else{
                            Faces[i].NameIndex = -1;    //a stranger
                            Faces[i].Color     =  1;
                        }
                    }

                    //test if the face is recognized, or should it be added to database
                    if(Faces[i].NameIndex == -1){
                        if(Faces[i].rect.height < MinHeightFace){
                            //a stranger with a small face
                            Faces[i].Color = 2; //too tiny
                        }
#ifdef AUTO_FILL_DATABASE
                        else{
                            //a stranger with a large face, who can be added to the database
                            if(FaceCnt>=MaxItemsDatabase){
                                cout << "Database full !" << endl;
                            }
                            else{
                                double blur=0.0;
#ifdef BLUR_FILTER_STRANGER
                                blur=Blur.Execute(aligned);
#endif // BLUR_FILTER_STRANGER
                                if(Warp.Angle<=MaxAngle && blur>=MaxBlur){
//                                cout <<"Probability : " << Faces[i].FaceProb << "   Angle : " << Warp.Angle << "   Blur : " << blur << endl;
//                                cv::imshow("Add",aligned);
                                    FaceCnt++;
                                    NewItemName = format("person %i",(int)FaceCnt);
                                    fc1.push_back(fc2);
                                    NameFaces.push_back(NewItemName);
                                    imwrite("./img/"+NewItemName+".jpg", aligned);
                                    cout << "Stored to database : " << NewItemName << endl;
                                }
                            }
                        }
#endif // AUTO_FILL_DATABASE
                    }
                }
            }
        }
#endif // RECOGNIZE_FACE

        Tend = chrono::steady_clock::now();

        DrawObjects(frame, Faces);

        //calculate frame rate
        f = chrono::duration_cast <chrono::milliseconds> (Tend - Tbegin).count();
        if(f>0.0) FPS[((Fcnt++)&0x0F)]=1000.0/f;
        for(f=0.0, i=0;i<16;i++){ f+=FPS[i]; }
        cv::putText(frame, cv::format("FPS %0.2f", f/16),cv::Point(10,20),cv::FONT_HERSHEY_SIMPLEX,0.6, cv::Scalar(180, 180, 0));

        //show output
        cv::imshow("Jetson Nano - 2014.5 MHz", frame);
        char esc = cv::waitKey(5);
        if(esc == 27) break;
    }

    cv::destroyAllWindows();

    return 0;
}
