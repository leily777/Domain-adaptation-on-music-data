# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 17:56:00 2018

@author: leila
"""

from PIL import Image, ImageOps
import numpy as np
from os import walk
import time
import random
import imageio

    
# For making train.txt for CatDog AlexNet finetuning
def generate_txt():
    
    path = "./trainCatDog"
    for (dirpath, dirnames, filenames) in walk(path):

        random.shuffle(filenames)
        size = len(filenames)
        trainlen = int(size*0.7)
        print (size,trainlen)

        myfile = open("trainer.txt","w")
        for i in range(trainlen):
            fpath = filenames[i]
            name  = fpath.split('.')[0]
            if name == 'cat':
                myfile.write(path +'/'+ fpath +' '+str(0) + '\n' )
            else:
                myfile.write(path +'/'+ fpath +' '+str(1) + '\n' )
        myfile.close()


        # Validater
        myfile = open("val.txt","w")
        nextiter = trainlen+int(size*0.15)        
        for i in range(trainlen,nextiter):
            fpath = filenames[i]
            name  = fpath.split('.')[0]
            if name == 'cat':
                myfile.write(path +'/'+ fpath +' '+str(0) + '\n' )
            else:
                myfile.write(path +'/'+ fpath +' '+str(1) + '\n' )
        myfile.close()



        # Tester
        myfile = open("tester.txt","w")
        nextiter = trainlen+int(size*0.15)        
        for i in range(nextiter,size):
            fpath = filenames[i]
            name  = fpath.split('.')[0]
            if name == 'cat':
                myfile.write(path +'/'+ fpath +' '+str(0) + '\n' )
            else:
                myfile.write(path +'/'+ fpath +' '+str(1) + '\n' )
        myfile.close()

def generate_sample(batch_size=64,widthw=51,heightw=51,   #batch_size should be factor of 4, due to class imbalance trick applied here
                    total_iterations = 1,test_flag = False,
                    pathimg="C:/Users/divy/Desktop/internship/Data-DomAdap-MusicScores/Einsiedeln1/SRC",
                    pathlbl="C:/Users/divy/Desktop/internship/Data-DomAdap-MusicScores/Einsiedeln1/GT"):
    
        if batch_size < 4:
            raise "batch_size inadequate"
    
        img = []
        for (dirpath, dirnames, filenames) in walk(pathimg):
            img.extend(filenames)
        src_images = []
        for i in range(len(img)-2):
            image = Image.open(dirpath + "/" + img[i])
            img_with_border = ImageOps.expand(image,border=25,fill='white')
            image = np.array(img_with_border) 
            height,width,ch =image.shape
            src_images.append(image)
        print("Use ",img[-1]," as validation-image")             
            
        imglbl = []
        for (dirpathlbl, dirnameslbl, filenameslbl) in walk(pathlbl):
            imglbl.extend(filenameslbl) 
        tgt_images = []
        for j in range(len(imglbl)-2):
            labelimage = np.load(dirpathlbl + "/" + imglbl[j])
            tgt_images.append(labelimage)
        print("Use ",filenameslbl[-1]," as validation-image")
        
        if test_flag:
            print ("Generating test samples")
            src_images = [src_images[-1]]
            tgt_images = [tgt_images[-1]]
        
        mydict = {0:[],1:[],2:[],3:[]}
        for k in range(len(tgt_images)):
            labelimage = tgt_images[k]
            image = src_images[k]
            #print (np.count_nonzero(labelimage))   only 12.6 percent of labels are not background, I believe equaly balaced training data would give better result
            heightlbl,widthlbl = labelimage.shape

            for i in range (0,heightlbl,25):#these step-size can be increased to reduce the class imbalance
                for j in range (0,widthlbl,25):
                    if image[i:i+heightw,j:j+widthw].shape == (heightw,widthw,3):
                        classlabel = labelimage[i,j]
                        mydict[classlabel].append((k,i,j))
                        
        print (len(mydict[0]),len(mydict[1]),len(mydict[2]),len(mydict[3]) )
        condition = True
        total_batchs = 0
        while condition:
            Smpl_label=[]
            Smpl=[]            
            for classlabel in range(0,4):
                samples = random.sample(mydict[classlabel], batch_size//4)
                for element in samples:
                    k,i,j = element
                    image = src_images[k]
                    Smpl.append(image[i:i+heightw,j:j+widthw])
                    Smpl_label.append(classlabel)

            Smpl_label = np.eye(4)[Smpl_label]    # one hot enconding
            yield (Smpl,Smpl_label)
            Smpl_label=[]
            Smpl=[]
            total_batchs += 1
            if total_batchs>=total_iterations:
                condition = False
        yield None,None
                        

if __name__ == '__main__':
    
    # For testing CatDog classification
    #generate_txt()


    batch_size = 512
    n_epoch = 1

    t0 = time.time()
    #Src= generate_sample(batch_size//2,widthw=227,heightw=227,
    #                pathimg="C:/Users/divy/Desktop/internship/Data-DomAdap-MusicScores/Einsiedeln1/SRC",
     #               pathlbl="C:/Users/divy/Desktop/internship/Data-DomAdap-MusicScores/Einsiedeln1/GT")
   
    Src = generate_sample(batch_size,227,227,5,True,pathimg="C:/Users/divy/Desktop/internship/Data-DomAdap-MusicScores/Salzinnes1/SRC",
                    pathlbl="C:/Users/divy/Desktop/internship/Data-DomAdap-MusicScores/Salzinnes1/GT")
    epoch=0
    t1 = t0
    counter=0
    myfile = open("tgt_tester.txt","w") 
    while epoch<n_epoch:
        print(counter)

        X_source, y_source = next(Src)
        if (X_source == None):
            Src= generate_sample(batch_size,227,227,5,True)
            X_source, y_source = next(Src)
            print ("epoch " +str(epoch)+ " finish")
            t1 = time.time()
            print("time is: ",t1-t0,"\n")
            t0 = t1
            epoch+=1             

        for i in range(len(X_source)):
            imageio.imwrite('./tgt_testset/sample'+str(counter)+'.png', X_source[i])
            myfile.write('./tgt_testset/sample'+str(counter)+'.png'+' '+str(np.argmax(y_source[i]) ) + '\n' )
            counter += 1
            if counter>=3000:
                epoch+=1
            
    myfile.close()
    print("Final time is: ",time.time()-t0,"\n")