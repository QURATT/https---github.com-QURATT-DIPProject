# import nest_asyncio
# nest_asyncio.apply()
# __import__('IPython').embed()
import sys
import cv2
import  numpy as np
import math
from skimage import io, util
import heapq
import matplotlib.pyplot as plt
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets,uic,QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QWidget,QStackedWidget
from PyQt5.QtGui import QPixmap
from skimage.transform import resize
# import logos


class WelcomeScreen(QDialog):
    def __init__(self):
        super(WelcomeScreen,self).__init__()
        uic.loadUi("window.ui",self)
        self.login.clicked.connect(self.gotologin)
        self.create.clicked.connect(self.gotosignup)
        self.textt.clicked.connect(self.gototexture)
        
    def gotologin(self):
        login = LoginScreen()
        widget.addWidget(login)
        widget.setCurrentIndex(widget.currentIndex()+1)
    
    def gotosignup(self):
        signup=SignupScreen()
        widget.addWidget(signup)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def gototexture(self):
        texture=TextureScreen()
        widget.addWidget(texture)
        widget.setCurrentIndex(widget.currentIndex()+1)

class TextureScreen(QDialog):
    def __init__(self):
        super(TextureScreen,self).__init__()
        loadUi("window3.ui",self) 
        # self.passedit.setEchoMode(QtWidgets.QLineEdit.Password)  
        # self.confirmpassedit.setEchoMode(QtWidgets.QLineEdit.Password)  
        # self.signupbut.clicked.connect(self.manageprofile)
        # self.back.clicked.connect(self.gotomain)
        self.back.clicked.connect(self.gotomain)
        # self.passedit.setEchoMode(QtWidgets.QLineEdit.Password) 
        self.login.clicked.connect(self.selectimage) 
    def selectimage(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.bmp)')
        global img
        # Load the selected image and store it in an array
        if file_path:
            target_image = cv2.imread(file_path).astype(np.float32)
            # img=cv2.resize(img,(0,0),None,0.5,0.5)
            self.img_array = np.array(target_image)
            # Display the image in the label
            pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(file_path))
            # self.image_label.setPixmap(pixmap)

            # normalize_img normalizes our output to be between 0 and 1
            def normalize_img(im):
                img = im.copy()
                img += np.abs(np.min(img))
                img /= np.max(img)
                return img

            #For defining patches dimension
            def l2_top_bottom(patch_top, patch_bottom, patch_curr, alpha, all_cm_blocks_target):
                block_top = patch_top[-overlap_size:, :]
                maxy = min(block_top.shape[1], block_size)    #Telling the maximum of opatches
                if patch_bottom.ndim == 3:
                    block_bottom = patch_bottom[:overlap_size]   #Assigning block bottom for overlap size
                    block_bottom = block_bottom[:, :maxy]     #Assigning block bottom for max size
                elif patch_bottom.ndim == 4:
                    block_bottom = patch_bottom[:, :overlap_size]      #Assigning block bottom for overlap size
                    block_bottom = block_bottom[:, :, :maxy]       #Assigning block bottom for max size
                else:
                    raise ValueError('patch_bottom must have 3 or 4 dimensions')       #Patch bottom must have 3 to 4 dimensions

                top_cost = alpha * l2_loss(block_top, block_bottom)
                curr_patch_intensities = np.sum(patch_curr, axis=-1)
                y2 = min(curr_patch_intensities.shape[0], block_size)    #Patch bottom must have 3 to 4 dimensions
                top_cost += (1 - alpha) * corr_loss(curr_patch_intensities[:y2, :], all_cm_blocks_target[:, :y2, :])

                return top_cost


            def l2_left_right(patch_left, patch_right, patch_curr, alpha, all_cm_blocks_target):
                block_left = patch_left[:, -overlap_size:]

                if patch_right.ndim == 3:
                    block_right = patch_right[:, :overlap_size]      #Assigning block right for overlap size
                elif patch_right.ndim == 4:
                    block_right = patch_right[:, :, :overlap_size]      #Assigning block bottom for overlap size
                else:
                    raise ValueError('patch_right must have 3 or 4 dimensions')

                # overlap error
                left_cost = alpha * l2_loss(block_left, block_right)
                # add correspondence error
                curr_patch_intensities = np.sum(patch_curr, axis=-1)           #Assigning block bottom for overlap size
                x2 = min(curr_patch_intensities.shape[1], block_size)
                left_cost += (1 - alpha) * corr_loss(curr_patch_intensities[:, :x2], all_cm_blocks_target[:, :, :x2])       #Calculating left cost

                return left_cost


            def corr_loss(block_1, block_2):
                return np.sum(np.sum((block_1 - block_2) ** 2, axis=-1), axis=-1)     #Returning np sum


            def l2_loss(block_1, block_2):
                sqdfs = np.sum((block_1 - block_2) ** 2, axis=-1)
                return np.sqrt(np.sum(np.sum(sqdfs, axis=-1), axis=-1))


            def select_min_patch(patches, cost, tolerance=0.1):         #Select minimum patch
                min_cost = np.min(cost)
                upper_cost_bound = min_cost + tolerance * min_cost
                # pick random patch within tolerance
                patch = patches[np.random.choice(np.argwhere(cost <= upper_cost_bound).flatten())]      #Return random flatten patches
                return patch


            def compute_error_surface(block_1, block_2):      #Compute error surface
                return np.sum((block_1 - block_2) ** 2, axis=-1)


            def min_vert_path(error_surf_vert):          #Select minimum vertical path
                top_min_path = np.zeros(block_size, dtype=np.int)
                top_min_path[0] = np.argmin(error_surf_vert[0, :], axis=0)
                for i in range(1, block_size):
                    err_mid_v = error_surf_vert[i, :]
                    mid_v = err_mid_v[top_min_path[i - 1]]

                    err_left = np.roll(error_surf_vert[i, :], 1, axis=0)
                    err_left[0] = np.inf
                    left = err_left[top_min_path[i - 1]]      #Select Top min path

                    err_right = np.roll(error_surf_vert[i, :], -1, axis=0)           #Select Surf Veretical
                    err_right[-1] = np.inf
                    right = err_right[top_min_path[i - 1]]

                    next_poss_pts_v = np.vstack((left, mid_v, right))               #Select Stack
                    new_pts_ind_v = top_min_path[i - 1] + (np.argmin(next_poss_pts_v, axis=0) - 1)
                    top_min_path[i] = new_pts_ind_v

                return top_min_path


            def min_hor_path(error_surf_hor):         #Select minimum horizontal path
                left_min_path = np.zeros(block_size, dtype=np.int)
                left_min_path[0] = np.argmin(error_surf_hor[:, 0], axis=0)
                for i in range(1, block_size):
                    err_mid_h = error_surf_hor[:, i]
                    mid_h = err_mid_h[left_min_path[i - 1]]            #Select minimum horizontal path

                    err_top = np.roll(error_surf_hor[:, i], 1, axis=0)
                    err_top[0] = np.inf
                    top = err_top[left_min_path[i - 1]]           #Select minimum horizontal path

                    err_bot = np.roll(error_surf_hor[:, i], -1, axis=0)           #Select minimum horizontal path
                    err_bot[-1] = np.inf
                    bot = err_bot[left_min_path[i - 1]]

                    next_poss_pts_h = np.vstack((top, mid_h, bot))        #Select minimum horizontal path
                    new_pts_ind_h = left_min_path[i - 1] + (np.argmin(next_poss_pts_h, axis=0) - 1)
                    left_min_path[i] = new_pts_ind_h

                return left_min_path


            def compute_lr_join(block_left, block_right, error_surf_vert=None):          #Returning left right join
                if error_surf_vert is None:
                    error_surf_vert = compute_error_surface(block_right, block_left)

                vert_path = min_vert_path(error_surf_vert)
                yy, xx = np.meshgrid(np.arange(block_size), np.arange(overlap_size))        #Returning left right join
                vert_mask = xx.T <= np.tile(np.expand_dims(vert_path, 1), overlap_size)

                lr_join = np.zeros_like(block_left)
                lr_join[:, :][vert_mask] = block_left[vert_mask]         #Returning block left
                lr_join[:, :][~vert_mask] = block_right[~vert_mask]        #Returning block right

                return lr_join


            def compute_bt_join(block_top, block_bottom, error_surf_hor=None):      #Returning left right join
                if error_surf_hor is None:
                    error_surf_hor = compute_error_surface(block_bottom, block_top)

                hor_path = min_hor_path(error_surf_hor)
                yy, xx = np.meshgrid(np.arange(block_size), np.arange(overlap_size))
                hor_mask = (xx.T <= np.tile(np.expand_dims(hor_path, 1), overlap_size)).T             #Returning left right join

                bt_join = np.zeros_like(block_top)
                bt_join[:, :][hor_mask] = block_top[hor_mask]
                bt_join[:, :][~hor_mask] = block_bottom[~hor_mask]   #Returning left right join

                return bt_join


            def lr_bt_join_double(best_left_block, right_block, best_top_block, bottom_block):    #Returning left bottom join
                error_surf_hor = compute_error_surface(best_left_block, right_block)

                maxy = min(bottom_block.shape[1], block_size)
                best_top_block = best_top_block[:, :maxy]
                error_surf_vert = compute_error_surface(best_top_block, bottom_block)     #Returning left right join

                vert_contrib = np.zeros_like(error_surf_vert)
                hor_contrib = np.zeros_like(error_surf_hor)

                vert_contrib[:, :overlap_size] += (error_surf_hor[:overlap_size, :] + error_surf_vert[:, :overlap_size]) / 2
                hor_contrib[:overlap_size, :] += (error_surf_vert[:, :overlap_size] + error_surf_hor[:overlap_size, :]) / 2

                error_surf_vert += vert_contrib
                error_surf_hor += hor_contrib

                left_right_join = compute_lr_join(right_block, best_left_block, error_surf_vert=error_surf_hor)    #Returning left right join
                bottom_top_join = compute_bt_join(bottom_block, best_top_block, error_surf_hor=error_surf_vert)

                return left_right_join, bottom_top_join


            def transfer_texture(texture_src, img_target, blk_size):   #Applying texture transfer
                h, w, c = texture_src.shape

                assert block_size < min(h, w)     #Returning left right join

                dh, dw = h * 2, w * 2

                nx_blocks = ny_blocks = max(dh, dw) // block_size     #Creating new blocks
                w_new = h_new = nx_blocks * blk_size - (nx_blocks - 1) * overlap_size

                img_target = resize(img_target, (h_new, w_new), preserve_range=True)     #Selecting targeted image
                target = img_target.copy()    #Making copy of target image

                n = 5
                for i in range(n):

                    osz = int(block_size / 6)

                    assert block_size < min(h, w)

                    y_max, x_max = h - block_size, w - block_size        #Selecting targeted image

                    xs = np.arange(x_max)
                    ys = np.arange(y_max)        #Selecting targeted image
                    all_blocks = np.array([texture_src[y:y + block_size, x:x + block_size] for x in xs for y in ys])
                    all_cm_blocks_target = np.sum(all_blocks, axis=-1)      #Selecting targeted image

                    img_target = resize(img_target, (h_new, w_new), preserve_range=True)
                    y_begin = 0
                    y_end = block_size

                    alpha_i = 0.8 * (i / (n - 1)) + 0.1        #Selecting targeted image

                    print('alpha = %.2f, block size = %d' % (alpha_i, block_size))
                    step = block_size - osz

                    for y in range(ny_blocks):          #Selecting targeted image

                        x_begin = 0
                        x_end = block_size

                        for x in range(nx_blocks):
                            if x == 0 and y == 0:
                                # randomly select top left patch
                                r = np.random.randint(len(all_blocks))
                                random_patch = all_blocks[r]
                                target[y_begin:y_end, x_begin:x_end] = random_patch

                                x_begin = x_end
                                x_end += step

                                continue

                            xa, xb = x_begin - block_size, x_begin     #Selecting targeted image
                            ya, yb = y_begin - block_size, y_begin

                            if y == 0:
                                y1 = 0
                                y2 = block_size
                                left_patch = target[y1:y2, xa:xb]     #Selecting targeted image
                                left_block = left_patch[:, -osz:]       #Selecting targeted image

                                current_patch = target[y2 - block_size:y2, x_end - block_size:x_end]

                                left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks,
                                                        patch_curr=current_patch, alpha=alpha_i,
                                                        all_cm_blocks_target=all_cm_blocks_target)
                                best_right_patch = select_min_patch(all_blocks, left_cost)
                                best_right_block = best_right_patch[:, :osz]

                                left_right_join = compute_lr_join(left_block, best_right_block)
                                # join left and right blocks
                                full_join = np.hstack(
                                    (target[y1:y2, xa:xb - osz], left_right_join, best_right_patch[:, osz:]))

                                xm = target[y1:y2, xa:x_end].shape[1]
                                target[y1:y2, xa:x_end] = full_join[:, :xm]
                            else:
                                if x == 0:
                                    x1 = 0
                                    x2 = block_size
                                    top_patch = target[ya:yb, x1:x2]     #Selecting targeted image
                                    top_block = top_patch[-osz:, :]        #Selecting targeted image

                                    current_patch = target[y_end - block_size:y_end, x2 - block_size:x2]

                                    top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks,
                                                            patch_curr=current_patch, alpha=alpha_i,
                                                            all_cm_blocks_target=all_cm_blocks_target)
                                    best_bottom_patch = select_min_patch(all_blocks, top_cost)        #Selecting targeted image
                                    best_bottom_block = best_bottom_patch[:osz, :]

                                    # join top and bottom blocks
                                    top_bottom_join = compute_bt_join(top_block, best_bottom_block)
                                    full_join = np.vstack(
                                        (target[ya:yb - osz, x1:x2], top_bottom_join, best_bottom_patch[osz:, :]))

                                    xm = target[ya:y_end, x1:x2].shape[1]
                                    target[ya:y_end, x1:x2] = full_join[:, :xm]
                                else:
                                    # overlap is L-shaped
                                    y1, y2 = y_begin - osz, y_end            #Selecting targeted image
                                    x1, x2 = x_begin - osz, x_end  #Selecting targeted image

                                    left_patch = target[y1:y2, xa:xb]
                                    top_patch = target[ya:yb, x1:x2]

                                    left_block = left_patch[:, -osz:]      # overlap is L-shaped
                                    top_block = top_patch[-osz:, :]

                                    current_patch = target[y2 - block_size:y2, x_end - block_size:x_end]

                                    left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks,
                                                            patch_curr=current_patch, alpha=alpha_i,
                                                            all_cm_blocks_target=all_cm_blocks_target)

                                    top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks,          # overlap is L-shaped
                                                            patch_curr=current_patch, alpha=alpha_i,
                                                            all_cm_blocks_target=all_cm_blocks_target)

                                    best_right_patch = best_bottom_patch = select_min_patch(all_blocks, top_cost + left_cost)

                                    best_right_block = best_right_patch[:, :osz]
                                    best_bottom_block = best_bottom_patch[:osz, :]

                                    left_right_join, top_bottom_join = lr_bt_join_double(best_right_block, left_block,
                                                                                        best_bottom_block, top_block)      # overlap is L-shaped
                                    # join left and right blocks
                                    full_lr_join = np.hstack(
                                        (target[y1:y2, xa:xb - osz], left_right_join, best_right_patch[:, osz:]))

                                    # join top and bottom blocks
                                    full_tb_join = np.vstack(
                                        (target[ya:yb - osz, x1:x2], top_bottom_join, best_bottom_patch[osz:, :])) # Stacking 

                                    target[ya:y_end, x1:x2] = full_tb_join
                                    target[y1:y2, xa:x_end] = full_lr_join

                            x_begin = x_end
                            x_end += step
                            if x_end > w_new:
                                x_end = w_new

                        y_begin = y_end
                        y_end += step

                        if y_end > h_new:
                            y_end = h_new

                return target


            def show_text_trans(img):
                plt.title('Texture Transfer')       #Displaying output
                plt.imshow(normalize_img(img))
                plt.axis('off')
                plt.show()


            source_texture = plt.imread('data/texture14.jpg').astype(np.float32)        #Taking input
            #target_image = plt.imread('data/man.jpg').astype(np.float32)

            block_size = 30
            overlap_size = int(block_size / 6)

            show_text_trans(transfer_texture(source_texture, target_image, block_size))

    def gotomain(self):
        welcome=WelcomeScreen()
        widget.addWidget(welcome)
        widget.setCurrentIndex(widget.currentIndex()+1)        
        
class LoginScreen(QDialog):
    def __init__(self):
        super(LoginScreen,self).__init__()
        loadUi("login.ui",self) 
        self.back.clicked.connect(self.gotomain)
        # self.passedit.setEchoMode(QtWidgets.QLineEdit.Password) 
        self.login.clicked.connect(self.selectimage)        # Sign in
    def selectimage(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.bmp)')
        global img
        # Load the selected image and store it in an array
        if file_path:
            input_img = cv2.imread(file_path).astype(np.float32)
            # img=cv2.resize(img,(0,0),None,0.5,0.5)
            self.img_array = np.array(input_img)
            # Display the image in the label
            pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(file_path))
            # self.image_label.setPixmap(pixmap)

            # normalize_img normalizes our output to be between 0 and 1
            def normalize_img(im):
                img = im.copy()
                img += np.abs(np.min(img))       # Normalizing
                img /= np.max(img)
                return img

            # Returning Top Bottom PAtches
            def l2_top_bottom(patch_top, patch_bottom):
                block_top = patch_top[-overlap_size:, :]

                if patch_bottom.ndim == 3:
                    block_bottom = patch_bottom[:overlap_size]
                elif patch_bottom.ndim == 4:
                    block_bottom = patch_bottom[:, :overlap_size]
                else:
                    raise ValueError('patch_right must have 3 or 4 dimensions')     #Have 3 or 4 dimensions

                top_cost = l2_loss(block_top, block_bottom)

                return top_cost

            # Returning Left Right Patches
            def l2_left_right(patch_left, patch_right):
                block_left = patch_left[:, -overlap_size:]

                if patch_right.ndim == 3:
                    block_right = patch_right[:, :overlap_size]
                elif patch_right.ndim == 4:
                    block_right = patch_right[:, :, :overlap_size]
                else:
                    raise ValueError('patch_right must have 3 or 4 dimensions')         #Must Have 3 or 4 dimensions

                left_cost = l2_loss(block_left, block_right)

                return left_cost


            def l2_loss(block_1, block_2):
                sqdfs = np.sum((block_1 - block_2) ** 2, axis=-1)
                return np.sqrt(np.sum(np.sum(sqdfs, axis=-1), axis=-1))       #Making blocks and comparing


            def select_min_patch(patches, cost):   #select min patch
                return patches[np.argmin(cost)]


            def select_min_patch_tol(patches, cost):     #Select min patch total
                min_cost = np.min(cost)
                tolerance = 0.1
                upper_cost_bound = min_cost + tolerance * min_cost
                # pick random patch within tolerance
                patch = patches[np.random.choice(np.argwhere(cost <= upper_cost_bound).flatten())]
                return patch


            def compute_error_surface(block_1, block_2):   #For computing errors
                return np.sum((block_1 - block_2) ** 2, axis=-1)


            def min_vert_path(error_surf_vert):     #Returning minimum vertical path
                top_min_path = np.zeros(block_size, dtype=np.int)
                top_min_path[0] = np.argmin(error_surf_vert[0, :], axis=0)
                for i in range(1, block_size):
                    err_mid_v = error_surf_vert[i, :]
                    mid_v = err_mid_v[top_min_path[i - 1]]

                    err_left = np.roll(error_surf_vert[i, :], 1, axis=0)          #Np roll Error left
                    err_left[0] = np.inf
                    left = err_left[top_min_path[i - 1]]

                    err_right = np.roll(error_surf_vert[i, :], -1, axis=0)              #Rolling error Right
                    err_right[-1] = np.inf
                    right = err_right[top_min_path[i - 1]]
       
                    next_poss_pts_v = np.vstack((left, mid_v, right))           #Stavking mid left right patches
                    new_pts_ind_v = top_min_path[i - 1] + (np.argmin(next_poss_pts_v, axis=0) - 1)
                    top_min_path[i] = new_pts_ind_v

                return top_min_path


            def min_hor_path(error_surf_hor):    #Returning minimum horizontal path            #Min horizontal path distance
                left_min_path = np.zeros(block_size, dtype=np.int)
                left_min_path[0] = np.argmin(error_surf_hor[:, 0], axis=0)
                for i in range(1, block_size):
                    err_mid_h = error_surf_hor[:, i]
                    mid_h = err_mid_h[left_min_path[i - 1]]

                    err_top = np.roll(error_surf_hor[:, i], 1, axis=0)
                    err_top[0] = np.inf
                    top = err_top[left_min_path[i - 1]]

                    err_bot = np.roll(error_surf_hor[:, i], -1, axis=0)
                    err_bot[-1] = np.inf
                    bot = err_bot[left_min_path[i - 1]]

                    next_poss_pts_h = np.vstack((top, mid_h, bot))
                    new_pts_ind_h = left_min_path[i - 1] + (np.argmin(next_poss_pts_h, axis=0) - 1)
                    left_min_path[i] = new_pts_ind_h

                return left_min_path


            def compute_lr_join(block_left, block_right, error_surf_vert=None):    #Computing left right join
                if error_surf_vert is None:
                    error_surf_vert = compute_error_surface(block_right, block_left)

                vert_path = min_vert_path(error_surf_vert)
                yy, xx = np.meshgrid(np.arange(block_size), np.arange(overlap_size))
                vert_mask = xx.T <= np.tile(np.expand_dims(vert_path, 1), overlap_size)

                lr_join = np.zeros_like(block_left)            #Computing left right join block left
                lr_join[:, :][vert_mask] = block_left[vert_mask]            #Computing left right join block left and vertical mask
                lr_join[:, :][~vert_mask] = block_right[~vert_mask]               #Computing left right join block left and vertical mask

                return lr_join


            def compute_bt_join(block_top, block_bottom, error_surf_hor=None):    #Returning block top and bottom path
                if error_surf_hor is None:
                    error_surf_hor = compute_error_surface(block_bottom, block_top)         #Computing left right join block left and vertical mask

                hor_path = min_hor_path(error_surf_hor)
                yy, xx = np.meshgrid(np.arange(block_size), np.arange(overlap_size))
                hor_mask = (xx.T <= np.tile(np.expand_dims(hor_path, 1), overlap_size)).T

                bt_join = np.zeros_like(block_top)
                bt_join[:, :][hor_mask] = block_top[hor_mask]             #Computing left right join block left and horizintal mask
                bt_join[:, :][~hor_mask] = block_bottom[~hor_mask]      #Computing left right join block left and horizintal mask

                return bt_join


            def lr_bt_join_double(best_left_block, right_block, best_top_block, bottom_block): 
                error_surf_hor = compute_error_surface(best_left_block, right_block)    #Returning best block top, right and bottom path
                error_surf_vert = compute_error_surface(best_top_block, bottom_block)

                vert_contrib = np.zeros_like(error_surf_vert)
                hor_contrib = np.zeros_like(error_surf_hor)

                vert_contrib[:, :overlap_size] += (error_surf_hor[:overlap_size, :] + error_surf_vert[:, :overlap_size]) / 2
                hor_contrib[:overlap_size, :] += (error_surf_vert[:, :overlap_size] + error_surf_hor[:overlap_size, :]) / 2

                error_surf_vert += vert_contrib
                error_surf_hor += hor_contrib

                left_right_join = compute_lr_join(right_block, best_left_block, error_surf_vert=error_surf_hor)
                bottom_top_join = compute_bt_join(bottom_block, best_top_block, error_surf_hor=error_surf_vert)

                return left_right_join, bottom_top_join


            def synth_texture_rand(texture, blk_size):     #Applying texture synthesis function
                h, w, c = texture.shape
                assert blk_size < min(h, w)

                y_max, x_max = h - blk_size, w - blk_size

                # desired size of new image is twice original one
                dh = h * 2
                dw = w * 2

                nx_blocks = ny_blocks = max(dh, dw) // blk_size
                w_new = h_new = nx_blocks * blk_size

                n_blocks = nx_blocks * ny_blocks

                texture_img = np.zeros((h_new, w_new, c), dtype=texture.dtype)

                # Choose random blocks
                xs = np.random.randint(0, x_max, size=n_blocks)
                ys = np.random.randint(0, y_max, size=n_blocks)
                ind = np.vstack((xs, ys)).T

                blocks = np.array([input_img[y:y + blk_size, x:x + blk_size] for x, y in ind])

                b = 0
                for y in range(ny_blocks):
                    for x in range(nx_blocks):
                        x1, y1 = x * blk_size, y * blk_size
                        x2, y2 = x1 + blk_size, y1 + blk_size
                        texture_img[y1:y2, x1:x2] = blocks[b]
                        b += 1

                return texture_img


            def synth_texture_neighborhood(texture, blk_size):    #Appllying on neighborhood pixels as well
                h, w, c = texture.shape

                assert blk_size < min(h, w)

                # desired size of new image is twice original one
                dh = h * 2
                dw = w * 2

                y_max, x_max = h - blk_size, w - blk_size
                nx_blocks = ny_blocks = max(dh, dw) // blk_size
                w_new = h_new = nx_blocks * blk_size - (nx_blocks - 1) * overlap_size

                xs = np.arange(x_max)
                ys = np.arange(y_max)
                all_blocks = np.array([input_img[y:y + blk_size, x:x + blk_size] for x in xs for y in ys])

                target_height = h_new
                target_width = w_new
                target = np.zeros((target_height, target_width, c), dtype=input_img.dtype)

                step = blk_size - overlap_size

                y_begin = 0
                y_end = blk_size

                for y in range(ny_blocks):

                    x_begin = 0
                    x_end = blk_size

                    for x in range(nx_blocks):
                        if x == 0 and y == 0:
                            # randomly select top left patch
                            r = np.random.randint(len(all_blocks))
                            random_patch = all_blocks[r]
                            target[y_begin:y_end, x_begin:x_end] = random_patch

                            x_begin = x_end
                            x_end += step

                            continue

                        xa, xb = x_begin - blk_size, x_begin
                        ya, yb = y_begin - blk_size, y_begin

                        if y == 0:
                            y1 = 0
                            y2 = blk_size

                            left_patch = target[y1:y2, xa:xb]
                            left_block = left_patch[:, -overlap_size:]
                            left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks)
                            best_right_patch = select_min_patch(all_blocks, left_cost)

                            best_right_block = best_right_patch[:, :overlap_size]

                            # join left and right blocks
                            left_right_join = np.hstack(
                                (left_block[:, :overlap_size // 2], best_right_block[:, overlap_size // 2:]))
                            full_join = np.hstack(
                                (target[y1:y2, xa:xb - overlap_size], left_right_join, best_right_patch[:, overlap_size:]))

                            target[y1:y2, xa:x_end] = full_join
                        else:
                            if x == 0:
                                x1 = 0
                                x2 = blk_size
                                top_patch = target[ya:yb, x1:x2]
                                top_block = top_patch[-overlap_size:, :]
                                top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks)
                                best_bottom_patch = select_min_patch(all_blocks, top_cost)
                                best_bottom_block = best_bottom_patch[:overlap_size, :]

                                # join top and bottom blocks
                                top_bottom_join = np.vstack(
                                    (top_block[:overlap_size // 2, :], best_bottom_block[overlap_size // 2:, :]))
                                full_join = np.vstack(
                                    (target[ya:yb - overlap_size, x1:x2], top_bottom_join, best_bottom_patch[overlap_size:, :]))

                                target[ya:y_end, x1:x2] = full_join
                            else:
                                # overlap is L-shaped
                                y1, y2 = y_begin - overlap_size, y_end
                                x1, x2 = x_begin - overlap_size, x_end

                                left_patch = target[y1:y2, xa:xb]
                                top_patch = target[ya:yb, x1:x2]

                                left_block = left_patch[:, -overlap_size:]
                                top_block = top_patch[-overlap_size:, :]

                                left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks)
                                top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks)

                                best_right_patch = best_bottom_patch = select_min_patch(all_blocks, top_cost + left_cost)

                                best_right_block = best_right_patch[:, :overlap_size]
                                best_bottom_block = best_bottom_patch[:overlap_size, :]

                                # join left and right blocks
                                left_right_join = np.hstack(
                                    (left_block[:, :overlap_size // 2], best_right_block[:, overlap_size // 2:]))
                                full_lr_join = np.hstack(
                                    (target[y1:y2, xa:xb - overlap_size], left_right_join, best_right_patch[:, overlap_size:]))

                                # join top and bottom blocks
                                top_bottom_join = np.vstack(
                                    (top_block[:overlap_size // 2, :], best_bottom_block[overlap_size // 2:, :]))
                                full_tb_join = np.vstack(
                                    (target[ya:yb - overlap_size, x1:x2], top_bottom_join, best_bottom_patch[overlap_size:, :]))

                                target[ya:y_end, x1:x2] = full_tb_join
                                target[y1:y2, xa:x_end] = full_lr_join

                        x_begin = x_end
                        x_end += step

                    y_begin = y_end
                    y_end += step

                return target


            def synth_texture(src_texture, blk_size):     #Final texture synthesis function
                h, w, c = src_texture.shape

                assert blk_size < min(h, w)

                y_max, x_max = h - blk_size, w - blk_size
                dh = h * 2
                dw = w * 2
                nx_blocks = ny_blocks = max(dh, dw) // blk_size
                w_new = h_new = nx_blocks * blk_size - (nx_blocks - 1) * overlap_size

                xs = np.arange(x_max)
                ys = np.arange(y_max)
                all_blocks = np.array([src_texture[y:y + blk_size, x:x + blk_size] for x in xs for y in ys])

                target_height = h_new
                target_width = w_new
                target = np.zeros((target_height, target_width, c), dtype=input_img.dtype)

                step = blk_size - overlap_size

                y_begin = 0
                y_end = blk_size

                for y in range(ny_blocks):

                    x_begin = 0
                    x_end = blk_size

                    for x in range(nx_blocks):
                        if x == 0 and y == 0:
                            # randomly select top left patch
                            r = np.random.randint(len(all_blocks))
                            random_patch = all_blocks[r]
                            target[y_begin:y_end, x_begin:x_end] = random_patch

                            x_begin = x_end
                            x_end += step

                            continue

                        xa, xb = x_begin - blk_size, x_begin
                        ya, yb = y_begin - blk_size, y_begin

                        if y == 0:
                            y1 = 0
                            y2 = blk_size

                            left_patch = target[y1:y2, xa:xb]
                            left_block = left_patch[:, -overlap_size:]
                            left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks)
                            best_right_patch = select_min_patch_tol(all_blocks, left_cost)
                            best_right_block = best_right_patch[:, :overlap_size]

                            left_right_join = compute_lr_join(left_block, best_right_block)
                            # join left and right blocks
                            full_join = np.hstack(
                                (target[y1:y2, xa:xb - overlap_size], left_right_join, best_right_patch[:, overlap_size:]))

                            target[y1:y2, xa:x_end] = full_join
                        else:
                            if x == 0:
                                x1 = 0
                                x2 = blk_size
                                top_patch = target[ya:yb, x1:x2]
                                top_block = top_patch[-overlap_size:, :]
                                top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks)
                                best_bottom_patch = select_min_patch_tol(all_blocks, top_cost)
                                best_bottom_block = best_bottom_patch[:overlap_size, :]

                                # join top and bottom blocks
                                top_bottom_join = compute_bt_join(top_block, best_bottom_block)
                                full_join = np.vstack(
                                    (target[ya:yb - overlap_size, x1:x2], top_bottom_join, best_bottom_patch[overlap_size:, :]))

                                target[ya:y_end, x1:x2] = full_join
                            else:
                                # overlap is L-shaped
                                y1, y2 = y_begin - overlap_size, y_end
                                x1, x2 = x_begin - overlap_size, x_end

                                left_patch = target[y1:y2, xa:xb]
                                top_patch = target[ya:yb, x1:x2]

                                left_block = left_patch[:, -overlap_size:]
                                top_block = top_patch[-overlap_size:, :]

                                left_cost = l2_left_right(patch_left=left_patch, patch_right=all_blocks)
                                top_cost = l2_top_bottom(patch_top=top_patch, patch_bottom=all_blocks)

                                best_right_patch = best_bottom_patch = select_min_patch_tol(all_blocks, top_cost + left_cost)

                                best_right_block = best_right_patch[:, :overlap_size]
                                best_bottom_block = best_bottom_patch[:overlap_size, :]

                                left_right_join, top_bottom_join = lr_bt_join_double(best_right_block, left_block,
                                                                                    best_bottom_block, top_block)
                                # join left and right blocks
                                full_lr_join = np.hstack(
                                    (target[y1:y2, xa:xb - overlap_size], left_right_join, best_right_patch[:, overlap_size:]))

                                # join top and bottom blocks
                                full_tb_join = np.vstack(
                                    (target[ya:yb - overlap_size, x1:x2], top_bottom_join, best_bottom_patch[overlap_size:, :]))

                                target[ya:y_end, x1:x2] = full_tb_join
                                target[y1:y2, xa:x_end] = full_lr_join

                        x_begin = x_end
                        x_end += step

                    y_begin = y_end
                    y_end += step

                return target


            def show_fig2a(texture_img):
                # plt.title('Figure 2a')
                # plt.imshow(normalize_img(texture_img))
                # plt.axis('off')
                # plt.show()
                pass


            def show_fig2b(texture_img):
                # plt.title('Figure 2b')
                # plt.imshow(normalize_img(texture_img))
                # plt.axis('off')
                # plt.show()
                pass


            def show_fig2c(texture_img):     #Showing final output
                plt.title('Figure 2c')
                plt.imshow(normalize_img(texture_img))
                plt.axis('off')
                plt.show()


            # texture_1 = plt.imread('data/texture1.jpg').astype(np.float32)

            block_size = 100
            overlap_size = int(block_size / 6)

            show_fig2a(synth_texture_rand(input_img, block_size))
            show_fig2b(synth_texture_neighborhood(input_img, block_size))
            show_fig2c(synth_texture(input_img, block_size))

        
        # cv2.imshow('ImageWindow', input_img)



    def gotomain(self):
        welcome=WelcomeScreen()
        widget.addWidget(welcome)
        widget.setCurrentIndex(widget.currentIndex()+1)     
        
class SignupScreen(QDialog):
    def __init__(self):
        super(SignupScreen,self).__init__()
        loadUi("window2.ui",self) 
        # self.passedit.setEchoMode(QtWidgets.QLineEdit.Password)  
        # self.confirmpassedit.setEchoMode(QtWidgets.QLineEdit.Password)  
        # self.signupbut.clicked.connect(self.manageprofile)
        # self.back.clicked.connect(self.gotomain)
        self.back.clicked.connect(self.gotomain)
        # self.passedit.setEchoMode(QtWidgets.QLineEdit.Password) 
        self.login.clicked.connect(self.selectimage) 
    def selectimage(self):
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Image', '', 'Images (*.png *.jpg *.bmp)')
        global img
        # Load the selected image and store it in an array
        if file_path:
            texture = cv2.imread(file_path).astype(np.float32)
            # img=cv2.resize(img,(0,0),None,0.5,0.5)
            self.img_array = np.array(texture)
            # Display the image in the label
            pixmap = QtGui.QPixmap.fromImage(QtGui.QImage(file_path))
            # self.image_label.setPixmap(pixmap)

            def randomPatch(texture, patchLength):    #Determining random patch
                h, w, _ = texture.shape
                i = np.random.randint(h - patchLength)
                j = np.random.randint(w - patchLength)
                return texture[i:i+patchLength, j:j+patchLength]

            def L2OverlapDiff(patch, patchLength, overlap, res, y, x):    #Determining L2 Overlap Different PAtches
                error = 0
                if x > 0:
                    left = patch[:, :overlap] - res[y:y+patchLength, x:x+overlap]
                    error += np.sum(left**2)
                if y > 0:
                    up   = patch[:overlap, :] - res[y:y+overlap, x:x+patchLength]
                    error += np.sum(up**2)
                if x > 0 and y > 0:
                    corner = patch[:overlap, :overlap] - res[y:y+overlap, x:x+overlap]
                    error -= np.sum(corner**2)

                return error

            def randomBestPatch(texture, patchLength, overlap, res, y, x):     #Determining Random Best PAtches
                h, w, _ = texture.shape
                errors = np.zeros((h - patchLength, w - patchLength))

                for i in range(h - patchLength):
                    for j in range(w - patchLength):
                        patch = texture[i:i+patchLength, j:j+patchLength]
                        e = L2OverlapDiff(patch, patchLength, overlap, res, y, x)
                        errors[i, j] = e

                i, j = np.unravel_index(np.argmin(errors), errors.shape)
                return texture[i:i+patchLength, j:j+patchLength]

            def minCutPath(errors):    #Defining minimum cut paths
                # dijkstra's algorithm vertical
                pq = [(error, [i]) for i, error in enumerate(errors[0])]
                heapq.heapify(pq)
                h, w = errors.shape
                seen = set()
                while pq:
                    error, path = heapq.heappop(pq)
                    curDepth = len(path)
                    curIndex = path[-1]
                    if curDepth == h:
                        return path
                    for delta in -1, 0, 1:
                        nextIndex = curIndex + delta
                        if 0 <= nextIndex < w:
                            if (curDepth, nextIndex) not in seen:
                                cumError = error + errors[curDepth, nextIndex]
                                heapq.heappush(pq, (cumError, path + [nextIndex]))
                                seen.add((curDepth, nextIndex))

            def minCutPath2(errors):
                # dynamic programming, unused
                errors = np.pad(errors, [(0, 0), (1, 1)], 
                                mode='constant', 
                                constant_values=np.inf)
                cumError = errors[0].copy()
                paths = np.zeros_like(errors, dtype=int)    
                for i in range(1, len(errors)):
                    M = cumError
                    L = np.roll(M, 1)
                    R = np.roll(M, -1)
                    # optimize with np.choose?
                    cumError = np.min((L, M, R), axis=0) + errors[i]
                    paths[i] = np.argmin((L, M, R), axis=0)   
                paths -= 1    
                minCutPath = [np.argmin(cumError)]
                for i in reversed(range(1, len(errors))):
                    minCutPath.append(minCutPath[-1] + paths[i][minCutPath[-1]]) 
                return map(lambda x: x - 1, reversed(minCutPath))

            def minCutPatch(patch, patchLength, overlap, res, y, x):
                patch = patch.copy()
                dy, dx, _ = patch.shape
                minCut = np.zeros_like(patch, dtype=bool)
                if x > 0:
                    left = patch[:, :overlap] - res[y:y+dy, x:x+overlap]
                    leftL2 = np.sum(left**2, axis=2)
                    for i, j in enumerate(minCutPath(leftL2)):
                        minCut[i, :j] = True
                if y > 0:
                    up = patch[:overlap, :] - res[y:y+overlap, x:x+dx]
                    upL2 = np.sum(up**2, axis=2)
                    for j, i in enumerate(minCutPath(upL2.T)):
                        minCut[:i, j] = True
                np.copyto(patch, res[y:y+dy, x:x+dx], where=minCut)
                return patch
            s = "https://raw.githubusercontent.com/axu2/image-quilting/master/"

            def quilt(texture, patchLength, numPatches, mode="cut", sequence=False):      #Defining quilting function
                texture = util.img_as_float(texture)
                overlap = patchLength // 6
                numPatchesHigh, numPatchesWide = numPatches
                h = (numPatchesHigh * patchLength) - (numPatchesHigh - 1) * overlap
                w = (numPatchesWide * patchLength) - (numPatchesWide - 1) * overlap
                res = np.zeros((h, w, texture.shape[2]))
                for i in range(numPatchesHigh):
                    for j in range(numPatchesWide):
                        y = i * (patchLength - overlap)
                        x = j * (patchLength - overlap)
                        if i == 0 and j == 0 or mode == "random":
                            patch = randomPatch(texture, patchLength)
                        elif mode == "best":
                            patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
                        elif mode == "cut":
                            patch = randomBestPatch(texture, patchLength, overlap, res, y, x)
                            patch = minCutPatch(patch, patchLength, overlap, res, y, x)          
                        res[y:y+patchLength, x:x+patchLength] = patch
                        if sequence:
                            io.imshow(res)
                            io.show()
                                 
                return res

            def quiltSize(texture, patchLength, shape, mode="cut"):    #Defining Quilt Size
                overlap = patchLength // 6
                h, w = shape
                numPatchesHigh = math.ceil((h - patchLength) / (patchLength - overlap)) + 1 or 1
                numPatchesWide = math.ceil((w - patchLength) / (patchLength - overlap)) + 1 or 1
                res = quilt(texture, patchLength, (numPatchesHigh, numPatchesWide), mode)
                return res[:h, :w]

            texture = io.imread(s+"test.png")
            # io.imshow(texture)
            # io.show()
            # io.close()
            # io.imshow(quilt(texture, 25, (6, 6), "random"))
            # io.show()
            # io.close()
            # io.imshow(quilt(texture, 25, (6, 6), "best"))
            # io.show()
            # io.close()
            # io.imshow(quilt(texture, 20, (6, 6), "cut"))
            # io.show()
            # io.close()

            io.imshow(quilt(texture, 20, (3, 3), "cut", True))
            io.show()
            # io.close()

    def gotomain(self):
        welcome=WelcomeScreen()
        widget.addWidget(welcome)
        widget.setCurrentIndex(widget.currentIndex()+1)  
    
    def manageprofile(self):
        user=self.usernameedit.text()
        password=self.passedit.text()
        confirmpass=self.confirmpassedit.text()
        
        
        if len(user)==0 or len(password)==0 or len(confirmpass)==0:
            self.error.setText("Please fill all fields")
            
        elif password!=confirmpass:
            self.error2.setText("Passwords donot match!!")
            
        else:
            manageprofile=ManageProfileScreen()
            widget.addWidget(manageprofile)
            widget.setCurrentIndex(widget.currentIndex()+1)
        
class ManageProfileScreen(QDialog):
    def __init__(self):
        super(ManageProfileScreen,self).__init__()
        loadUi("manageprofile.ui",self)
        self.accbutton.clicked.connect(self.gotoactions)
        self.profilepic.clicked.connect(self.gotoimageview)
        
    def gotoimageview(self):
        self.label=self.findChild(QLabel,"profileimage")
        
    def gotoactions(self):
        actions=ActionScreen()
        widget.addWidget(actions)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
class FeedbackScreen(QDialog):
    def __init__(self):
        super(FeedbackScreen,self).__init__()
        loadUi("feedback.ui",self)
        self.back.clicked.connect(self.gotomain)
    def gotomain(self):
        actions=ActionScreen()
        widget.addWidget(actions)
        widget.setCurrentIndex(widget.currentIndex()+1)
        
class  ActionScreen(QDialog):
      def __init__(self):
        super(ActionScreen,self).__init__()
        loadUi("actions.ui",self)
        self.inputvideo.clicked.connect(self.gotoinputscreen)
        self.viewfeedback.clicked.connect(self.gotofeedbackpage)
      def gotoinputscreen(self):
          inputscreen=InputScreen()
          widget.addWidget(inputscreen)
          widget.setCurrentIndex(widget.currentIndex()+1)
      def gotofeedbackpage(self):
          feedback=FeedbackScreen()
          widget.addWidget(feedback)
          widget.setCurrentIndex(widget.currentIndex()+1) 
          

class InputScreen(QDialog):
    def __init__(self):
        super(InputScreen,self).__init__()
        loadUi("videoinput.ui",self)
        self.back.clicked.connect(self.gotomain)
    def gotomain(self):
        actions=ActionScreen()
        widget.addWidget(actions)
        widget.setCurrentIndex(widget.currentIndex()+1) 
    
           
          
        
app=QApplication(sys.argv)
widget=QStackedWidget()
welcome=WelcomeScreen()
widget.addWidget(welcome)  
widget.setFixedHeight(500)
widget.setFixedWidth(700)
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")


    

    

           
    