from data_collect import *
from feature_extraction import *
from window_search import *
from classifier import *

import cv2
from scipy.ndimage.measurements import label
# import imageio
# imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


cspace='YUV'
pix_per_cell=16
cell_per_block=2
orient=9
hog_channel='All'
spatial_size=(32, 32)
hist_bins=32
hist_range=(0, 256)
spatial_feat=True
hist_feat=True
hog_feat=True

# Collect Data
car_image_paths, noncar_image_paths = collect_data()
# Split Train/Test Dataset
X_train,X_test,y_train,y_test,X_scaler = dataset_split(car_image_paths,noncar_image_paths,
											cspace,pix_per_cell,cell_per_block,orient,
											hog_channel,spatial_size,hist_bins,hist_range,
											spatial_feat,hist_feat,hog_feat)
# Train Linear SVC
svc = linearSVC_Classifier(X_train,y_train,X_test,y_test)

def process_image(image):
	span_ratio1,span_ratio2,span_ratio3,span_ratio4 = 1.25,1.25,1,1
	wsize1,wsize2,wsize3,wsize4 = 64, 128, 192, 256
	ystart1,ystart2,ystart3,ystart4 = 400, 390, 380, 370
	overlap1,overlap2,overlap3,overlap4 = 0.75,0.75,0.75,0.75
	x_start_stop=[[None,None],[None,None],[None,None],[None,None]]
	y_start_stop=[[ystart1,ystart1+wsize1*span_ratio1],[ystart2,ystart2+wsize2*span_ratio2],[ystart3,ystart3+wsize3*span_ratio3],[ystart4,ystart4+wsize4*span_ratio4]]
	xy_window=[(wsize1,wsize1),(wsize2,wsize2),(wsize3,wsize3),(wsize4,wsize4)]
	xy_overlap=[(overlap1,overlap1),(overlap2,overlap2),(overlap3,overlap3),(overlap4,overlap4)]

	hot_windows = []
	for i in range(len(x_start_stop)):
		window_list = sliding_window(image,x_start_stop[i],y_start_stop[i],xy_window[i],xy_overlap[i])
		hot_windows.append(search_windows(image,window_list,svc,X_scaler,cspace,pix_per_cell,
						cell_per_block,orient,hog_channel,spatial_size,hist_bins,
						hist_range,spatial_feat,hist_feat,hog_feat))

	hot_windows = [item for sublist in hot_windows for item in sublist]

	heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
	heatmap = add_heat(heatmap,hot_windows)
	heatmap = apply_threshold(heatmap,3)
	# heatmap = np.clip(heat,0,255)
	labels = label(heatmap)
	window_image = draw_labeled_boxes(np.copy(image),labels)

	return window_image

if not os.path.exists('test_videos_output/'):
    os.makedirs('test_videos_output/')
video_output1= 'test_videos_output/processed_project_video.mp4'
# video_output2 = 'test_videos_output/processed_challenge_video.mp4'
clip1 = VideoFileClip("project_video.mp4")
# clip2 = VideoFileClip("challenge_video.mp4")
video_process = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# %time white_clip.write_videofile(white_output, audio=False)
video_process.write_videofile(video_output1, audio=False)
