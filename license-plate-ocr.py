import sys
import cv2
import numpy as np
import traceback
import time

import darknet.python.darknet as dn

from os.path 				import splitext, basename
from glob					import glob
from darknet.python.darknet import detect
from src.label				import dknet_label_conversion
from src.utils 				import nms


if __name__ == '__main__':

	try:
	
		input_dir  = '/root/alpr-unconstrained-master/samples/12_13_mid'
		output_dir = input_dir

		ocr_threshold = .4

		ocr_weights = 'data/ocr/yolov3-LP_40000.weights'
		ocr_netcfg  = 'data/ocr/yolov3-LP.cfg_train'
		ocr_dataset = 'data/ocr/LP.data'

		ocr_net  = dn.load_net(ocr_netcfg.encode('utf-8'), ocr_weights.encode('utf-8'), 0)
		ocr_meta = dn.load_meta(ocr_dataset.encode('utf-8'))
		imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

		print('Performing OCR...')
		start = time.clock()
		for i,img_path in enumerate(imgs_paths):

			print('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])

			R,(width,height) = detect(ocr_net, ocr_meta, img_path.encode('utf-8') ,thresh=ocr_threshold, nms=None)

			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)
				sum_h = []
				for l in L:
					sum_h.append(l.tl()[1])
				
				mean = np.mean(sum_h)
				var = np.var(sum_h)
				if var<0.001:
					L.sort(key=lambda x: x.tl()[0])       #one layer
				else:
					L_top = []
					L_bot = []
					for l in L:
						if l.tl()[1]<mean:
							L_top.append(l)
						elif l.tl()[1]>=mean:
							L_bot.append(l)
					L_top.sort(key=lambda x: x.tl()[0])
					L_bot.sort(key=lambda x: x.tl()[0])
					L = L_top + L_bot
				#L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])

				with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					f.write(lp_str + '\n')

				print('\t\tLP: %s' % lp_str)

			else:

				print('No characters found')
		end = time.clock()
		print("\ttime: %f s" % (end-start))
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
