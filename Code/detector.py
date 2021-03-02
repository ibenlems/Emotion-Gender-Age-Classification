#############  dependancies  ############# 
import cv2
import numpy as np
import mtcnn
from facenet_pytorch import MTCNN
from keras.models import load_model 
import PIL
import sys
import time
import imutils
from termcolor import colored
sys.stderr = sys.stdout


#############  Global variables  ############# 
gender = ['Woman','Man']
emotion_dict = {
    0: 'Angry',
    1: 'Disgust', 
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}

#############  Arguments  ############# 
args = sys.argv

#############  load face detectors   ############# 
detector = mtcnn.MTCNN()
try :
  detector_py = MTCNN(select_largest=False,device='cuda')
except :
  print(colored("<<<<<<<<<<<<< No GPU found >>>>>>>>>>>>>",'green'))
  detector_py = MTCNN()
facecasc =cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


############# load models #############
gender_model = load_model('./Models/simple_CNN.81-0.96.hdf5',compile=False)

emotion_model = load_model('./Models/emotion_model2.h5')


############# Main Functions #############
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def detect_face_py(img):
    boxes , probs = detector_py.detect(img)
    return_res = []
    
    if boxes is None :
      return return_res
    for boxe in boxes:
        x, y, width, height = boxe
        width , height = width-x, height-y
        center = [x+(width/2), y+(height/2)]
        max_border = int(max(width, height))

        # center alignment
        left = max(int(center[0]-(max_border/2)), 0)
        right = max(int(center[0]+(max_border/2)), 0)
        top = max(int(center[1]-(max_border/2)), 0)
        bottom = max(int(center[1]+(max_border/2)), 0)

        # crop the face
        center_img_k = img[top:top+max_border, 
                           left:left+max_border, :]

        age_preds = 23

        ####### Gender #########
        center_im_geneder = np.array(cv2.resize(center_img_k,(gender_model.input_shape[1:3])))
        center_im_geneder = center_im_geneder.astype('float32')
        center_im_geneder /=255
        center_im_geneder = np.expand_dims(center_im_geneder, 0)
        gender_preds = gender_model.predict(center_im_geneder)

        ####### Emotion #########
        grey_img = np.array(PIL.Image.fromarray(center_img_k).resize([48, 48]))
        img_emotions = rgb2gray(grey_img).reshape(1, 48, 48, 1)       
        img_emotions /= 255
        emotion_preds = emotion_model.predict(img_emotions)
        
        # output to the cv2
        return_res.append([top, right, bottom, left, gender_preds, age_preds, emotion_preds])
        
    return return_res


def detect_face(img,use_mtcnn=True):

    if use_mtcnn :
      faces = detector.detect_faces(img)
    else :
      faces = facecasc.detectMultiScale(img,scaleFactor=1.3, minNeighbors=5)

    return_res = []
    
    for face in faces:
        if use_mtcnn :
          x, y, width, height = face['box']
        else :
          x, y, width, height = face
        center = [x+(width/2), y+(height/2)]
        max_border = max(width, height)

        # center alignment
        left = max(int(center[0]-(max_border/2)), 0)
        right = max(int(center[0]+(max_border/2)), 0)
        top = max(int(center[1]-(max_border/2)), 0)
        bottom = max(int(center[1]+(max_border/2)), 0)

        # crop the face
        center_img_k = img[top:top+max_border, 
                           left:left+max_border, :]

        age_preds = 23

        ####### Gender #########
        center_im_geneder = np.array(cv2.resize(center_img_k,(gender_model.input_shape[1:3])))
        center_im_geneder = center_im_geneder.astype('float32')
        center_im_geneder /=255
        center_im_geneder = np.expand_dims(center_im_geneder, 0)
        gender_preds = gender_model.predict(center_im_geneder)

        ####### Emotion #########
        grey_img = np.array(PIL.Image.fromarray(center_img_k).resize([48, 48]))
        img_emotions = rgb2gray(grey_img).reshape(1, 48, 48, 1)       
        img_emotions /= 255
        emotion_preds = emotion_model.predict(img_emotions)
        
        # output to the cv2
        return_res.append([top, right, bottom, left, gender_preds, age_preds, emotion_preds])
        
    return return_res
  
  
  
############# detection using webcam #############
def webcam_detector() :
	# Get a reference to webcam 
	video_capture = cv2.VideoCapture(0)


	while True:
    		# Grab a single frame of video
    		ret, frame = video_capture.read()

    		if not ret:
      			break
    		# Convert the image from BGR color (which OpenCV uses) to RGB color 
    		rgb_frame = frame[:, :, ::-1]

    		# Find all the faces in the current frame of video
    		face_locations = detect_face_py(rgb_frame)

    		# Display the results
    		for top, right, bottom, left, gender_preds, age_preds, emotion_preds in face_locations:
        		# Draw a box around the face
        		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        		gender_text = gender[np.argmax(gender_preds)]
        		cv2.putText(frame, 'Gender: {}({:.3f})'.format(gender_text, np.max(gender_preds)), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        		cv2.putText(frame, 'Age: {}'.format(age_preds), (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        		cv2.putText(frame, 'Emotion: {}({:.3f})'.format(emotion_dict[np.argmax(emotion_preds)], np.max(emotion_preds)), (left, top-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    		# Display the resulting image
    		cv2.imshow('Video', frame)

    		# Hit 'q' on the keyboard to quit!
    		if cv2.waitKey(1) & 0xFF == ord('q'):
        		break

	# Release handle to the webcam
	video_capture.release()
	cv2.destroyAllWindows()
	
	return
	

############# detection within images #############
def image_detector():
	path_to_image=args[2]
	image_output=args[3]
	# load our input image and grab its spatial dimensions
	image = cv2.imread(path_to_image,cv2.COLOR_BGR2RGB)
	if image is None:
		print(colored('############# input_path argument is invalid #############','red'))
		print("try 'detector help' for more information")
		return
	face_locations = detect_face_py(image)

	# Display the results
	for top, right, bottom, left, gender_preds, age_preds, emotion_preds in face_locations:
        	# Draw a box around the face
        	cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        
        	gender_text = gender[np.argmax(gender_preds)]
        	cv2.putText(image, 'Gender: {}({:.3f})'.format(gender_text, np.max(gender_preds)), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        	cv2.putText(image, 'Age: {}'.format(age_preds), (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        	cv2.putText(image, 'Emotion: {}({:.3f})'.format(emotion_dict[np.argmax(emotion_preds)], np.max(emotion_preds)), (left, top-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

	cv2.imshow('image',image)
	cv2.waitKey(0)
	try :
		cv2.imwrite(image_output, image)
	except :
		print(colored('############# output_path argument is invalid #############','red'))
		print("try 'detector help' for more information")
	
	return
	


############# detection within videos #############
def video_detector(stride=1):
	path_to_video=args[2]
	output_path=args[3]
	try :
		stride = int(args[4])
	except :
		pass
	
	print(colored("<<<<<<<<<<<<<< stride == {} >>>>>>>>>>>>>>".format(stride),'green'))
	# Get a reference to webcam 
	video_capture = cv2.VideoCapture(path_to_video)
	#initialize the writer
	writer = None

	# try to determine the total number of frames in the video file
	try:
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
			else cv2.CAP_PROP_FRAME_COUNT
		total = int(video_capture.get(prop))
		print(colored("<<<<<<<<<<< {} total frames in video >>>>>>>>>>>".format(total),'green'))

	# an error occurred while trying to determine the total
	# number of frames in the video file
	except:
		print(colored("############# could not determine # of frames in this video #############",'red'))
		print(colored("############# no approx. completion time can be provided #############",'red'))
		total = -1
	
	#frame number
	i = 0
	start = time.time()
	while True:
    		# Grab a single frame of video
    		ret, frame = video_capture.read()

    		if not ret:
      			break
    		# Convert the image from BGR color (which OpenCV uses) to RGB color 
    		rgb_frame = frame[:, :, ::-1]

    		if i%stride==0 :
        		# Find all the faces in the current frame of video
        		start_frame = time.time()
        		face_locations = detect_face_py(rgb_frame)
        		end_frame = time.time()

    		# Display the results
    		for top, right, bottom, left, gender_preds, age_preds, emotion_preds in face_locations:
        		# Draw a box around the face
        		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        		gender_text = gender[np.argmax(gender_preds)]
        		cv2.putText(frame, 'Gender: {}({:.3f})'.format(gender_text, np.max(gender_preds)), (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        		cv2.putText(frame, 'Age: {}'.format(age_preds), (left, top-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
        		cv2.putText(frame, 'Emotion: {}({:.3f})'.format(emotion_dict[np.argmax(emotion_preds)], np.max(emotion_preds)), (left, top-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

   		# check if the video writer is None
    		if writer is None :
  		      # initialize our video writer
		      fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		      writer = cv2.VideoWriter(output_path, fourcc, 30,(frame.shape[1], frame.shape[0]), True)
		      # some information on processing single frame
		      if total > 0:
			          elap = (end_frame - start_frame)
			          print(colored("<<<<<<<<<<< single frame took {:.4f} seconds >>>>>>>>>>>".format(elap),'green'))
			          print(colored("<<<<<<<<<<< total time estimated to finish processing: {:.4f}s >>>>>>>>>>>".format(elap * total/stride),'green'))

    
    		# write the output frame to disk
    		writer.write(frame)
    		i+=1

	# Release 
	video_capture.release()
	if writer is None :
		print(colored('############# input_path argument is invalid #############','red'))
		print("try 'detector help' for more information")
		return
	writer.release()

	#print time spent
	end = time.time()
	print(colored('<<<<<<<<<<<<< Total time spent: {}s >>>>>>>>>>>>>'.format(end-start),'green'))
	return





############### Execute Main function ############
if args[1]=='image' :
	image_detector()
elif args[1]=='video' :
	video_detector()
elif args[1]=='webcam' :
	webcam_detector()
else :
	print("Wrong arguments were given, Try 'detector help' for more information ")
	


