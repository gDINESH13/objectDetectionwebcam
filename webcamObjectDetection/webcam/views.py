from django.http.response import StreamingHttpResponse
from django.shortcuts import render
import cv2
import threading
from django.views.decorators import gzip

# Create your views here.
from .videotrial import detection


#to capture video class
class VideoCamera(object):
    def __init__(self) :
        self.video=cv2.VideoCapture(0)
        self.grabbed,self.frame=self.video.read()
        threading.Thread(target=self.update,args=()).start()
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        image=self.frame
        result=detection(image)
        _,jpeg=cv2.imencode('.jpg',result)
        return jpeg.tobytes()

    def update(self):
        while True:
            self.grabbed,self.frame=self.video.read()

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield (b'--frame\r\n' 
        b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')

@gzip.gzip_page
def Home(request):
    try:
        cam=VideoCamera()
        return StreamingHttpResponse(gen(cam),content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request,'webcam/camera.html')