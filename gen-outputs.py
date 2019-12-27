import sys
import cv2
import numpy as np

from glob						import glob
from os.path 					import splitext, basename, isfile
from src.utils 					import crop_region, image_files_from_folder
from src.drawing_utils			import draw_label, draw_losangle, write2img
from src.label 					import lread, Label, readShapes

from pdb import set_trace as pause


YELLOW = (  0,255,255)
RED    = (  0,  0,255)

input_dir = '/root/alpr-unconstrained-master/samples/12_15'#sys.argv[1]
process_dir = '/root/alpr-unconstrained-master/samples/12_15_mid'
output_dir ='/root/alpr-unconstrained-master/samples/12_15_op' #sys.argv[2]

img_files = image_files_from_folder(input_dir)

for img_file in img_files:

	bname = splitext(basename(img_file))[0]

	I = cv2.imread(img_file)

	sys.stdout.write('%s' % bname)



	#draw_label(I,lcar,color=YELLOW,thickness=3)

	lp_label 		= '%s/%s_lp.txt'		% (process_dir,bname)
	lp_label_str 	= '%s/%s_lp_str.txt'	% (process_dir,bname)

	if isfile(lp_label):

		Llp_shapes = readShapes(lp_label)
		pts = Llp_shapes[0].pts#.reshape(2,1) + lcar.tl().reshape(2,1)
		ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
		draw_losangle(I,ptspx,RED,3)

		if isfile(lp_label_str):
			with open(lp_label_str,'r') as f:
				lp_str = f.read().strip()
				llp = Label(0,tl=pts.min(1),br=pts.max(1))
				I = write2img(I,llp,lp_str)

				sys.stdout.write(',%s' % lp_str)

	cv2.imwrite('%s/%s_output.png' % (output_dir,bname),I)
	sys.stdout.write('\n')


