import numpy as np
import shutil
import os
import glob
import sys
import random


def choose_image_only(fl):
    '''
    This function returns true if the file is a image file.
    '''
    ext = (".jpg", ".jpeg", ".JPG", ".png")
    return fl.endswith(ext)
def choose_random_img(s_lst, n_test = 0.1, n_val = 0.2):
    '''
    Randomly choose test images for each species
    
    s_lst : List of images of a particular species
    n_species : Percentage (specified as float) to be selected as test
    '''
    img_filenames = s_lst
    test_files = random.sample(img_filenames, int(n_test*len(s_lst)))
    temp_files = [x for x in img_filenames if x not in test_files]
    val_files = random.sample(temp_files, int(n_val*len(s_lst)))
    train_files = [x for x in temp_files if x not in val_files]
    return (train_files, val_files, test_files)

def make_img_path(lst):
    '''
    Function to make paths for images
    '''
    return map(lambda x: lst[0]+'\\'+ x, lst[1])

def make_dirtree(src, dest):
    '''
    Function to copy folder structure without files
    '''
    for dirpath, dirnames, filenames in os.walk(src):
        structure = os.path.join(dest, dirpath[len(src):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits! = "+structure)
    return None

def copy_images(tup):
    src,dst = tup
    shutil.copyfile(src, dst)
    return None

def make_train_test(folder):
    '''
    folder = parent_folder
    file system : *nix stored in the following structure
    parent folder(work/)
            |-- grayscale_consol/
            |     |-- <species 1>/
            |     |      |- images.jpg
            |     |-- <species 2>/ 
            |            |- images.jpg
            |           .
            |           .
            |-- train/
            |     |-- <species 1>/
            |     |      |- images.jpg
            |     |-- <species 2>/
            |            |- images.jpg
            |           .
            |           . 
            |-- val/
            |     |-- <species 1>/
            |     |      |- images.jpg
            |     |-- <species 2>/
            |            |- images.jpg
            |           .
            |           .                          
            |-- test/
            |     |-- <species 1>/
            |     |      |- images.jpg
            |     |-- <species 2>/
            |            |- images.jpg        
            |           .
            |           .        
    '''

    all_dat = folder + 'CNN-Total/'
    train = folder + 'Train/'
    val = folder + 'Validate/'
    test = folder + 'Test/'
    
    n_test = 0.1
    n_val = 0.2
    
    try:
        shutil.rmtree(train)
    except:
        pass    
    try:
        shutil.rmtree(val)
    except:
        pass 
    try:
        shutil.rmtree(test)
    except:
        pass 
    
    os.mkdir(train)
    os.mkdir(val)
    os.mkdir(test)
    
    make_dirtree(all_dat, train)
    make_dirtree(all_dat, val)
    make_dirtree(all_dat, test)
    
    species_list = glob.glob(all_dat +'*')
    gen_path = map(os.walk, species_list)
    lst_i = map(lambda x: x.next(), gen_path)
    lst_a = np.array(lst_i, dtype = object)
    
    
    combined_list = map(lambda x: choose_random_img(x, n_test, n_val), lst_a[:,2])    
    combined_arr = np.array(combined_list)
    
    train_list = combined_arr[:,0]
    val_list = combined_arr[:,1]
    test_list = combined_arr[:,2]
    
    train_img_srcpaths = map(lambda ex : make_img_path(ex) , zip(lst_a[:,0], list(train_list)))
    train_src = reduce(lambda x,y: x+y,train_img_srcpaths)
    train_src_dest = map(lambda x : (x,x.replace(all_dat, train)), train_src)    

    
    val_img_srcpaths = map(lambda ex : make_img_path(ex) , zip(lst_a[:,0], list(val_list)))
    val_src = reduce(lambda x,y: x+y,val_img_srcpaths)
    val_src_dest = map(lambda x : (x,x.replace(all_dat, val)), val_src)    

    
    test_img_srcpaths = map(lambda ex : make_img_path(ex) , zip(lst_a[:,0], list(test_list)))
    test_src = reduce(lambda x,y: x+y,test_img_srcpaths)
    test_src_dest = map(lambda x : (x,x.replace(all_dat, test)), test_src)
      
    no_res = map(copy_images, train_src_dest)
    no_res1 = map(copy_images, test_src_dest)
    no_res2 = map(copy_images, val_src_dest)
    return None     

def main(work_folder):
    make_train_test(work_folder)


if __name__ == '__main__':
    work_folder = sys.argv[1]
    main(work_folder)