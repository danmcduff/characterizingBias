from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime
import json
import requests
import urllib.request
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from azure.storage.blob import BlockBlobService 
from watson_developer_cloud import VisualRecognitionV3
from watson_developer_cloud import WatsonApiException
from google.cloud import vision
from google.protobuf.json_format import MessageToDict

import numpy
import scipy.misc as misc
import cv2
import argparse
import pprint
import sys
import os
sys.path.insert(0, './')

import tensorflow as tf
import configWrapper as config
import tfutil
import dataset
import misc
import util_scripts_wrapper as util

import boto3

from sightengine.client import SightengineClient

class UserFun():
    def __init__(self, api, save_dir):
        ########################
        # INITIALIZE MODEL:
        ########################
       
        misc.init_output_logging()
        numpy.random.seed(config.random_seed)
        print('Initializing TensorFlow...')
        os.environ.update(config.env)
        tfutil.init_tf(config.tf_config)
        #-----------------
        network_pkl = misc.locate_network_pkl(14, None)
        print('Loading network from "%s"...' % network_pkl)
        self.G, self.D, self.Gs = misc.load_network_pkl(14, None)    
        self.random_state = numpy.random.RandomState()

        ########################
        # INITIALIZE API INFORMATION:
        ########################
        # Azure storage information
        self.table_account_name = ''        # NEEDS TO BE COMPLETED WITH AZURE ACCOUNT
        self.table_account_key = ''         # NEEDS TO BE COMPLETED WITH AZURE ACCOUNT
        self.block_blob_service = BlockBlobService(self.table_account_name, self.table_account_key) 
        self.https_prefix = ''              # NEEDS TO BE COMPLETED WITH AZURE ACCOUNT
        self.table_service = TableService(account_name=self.table_account_name, account_key=self.table_account_key)
        self.container_name = ''            # NEEDS TO BE COMPLETED WITH AZURE ACCOUNT

        # Microsoft face detection 
        self.msft_face_detection_key = ''  # NEEDS TO BE COMPLETED WITH MSFT API ACCOUNT
        self.msft_face_detection_url = ''  # NEEDS TO BE COMPLETED WITH MSFT API ACCOUNT
        self.msft_face_attributes = 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
        self.msft_headers = {'Ocp-Apim-Subscription-Key': self.msft_face_detection_key}
        self.msft_params = {
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': self.msft_face_attributes
        }

        # FacePlusPlus face detection 
        self.faceplusplus_http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
        self.faceplusplus_key = ""          # NEEDS TO BE COMPLETED WITH FACE++ API ACCOUNT
        self.faceplusplus_secret = ""       # NEEDS TO BE COMPLETED WITH FACE++ API ACCOUNT

        # IBM Watson face detection
        self.IBM_visual_recognition = VisualRecognitionV3(
            version='2018-03-19',
            iam_api_key='',                 # NEEDS TO BE COMPLETED WITH IBM API ACCOUNT
            url = 'https://gateway.watsonplatform.net/visual-recognition/api'
        )
        
        # Amazon AWS Rekognition face detection:
        self.amazon_face_detection_id = ''  # NEEDS TO BE COMPLETED WITH AMAZON API ACCOUNT
        self.amazon_face_detection_key = '' # NEEDS TO BE COMPLETED WITH AMAZON API ACCOUNT
        self.amazon_client = boto3.client('rekognition','us-east-1',aws_access_key_id=self.amazon_face_detection_id,aws_secret_access_key=self.amazon_face_detection_key)

        # SightEngine:
        self.SEclient = SightengineClient('', '')   # NEEDS TO BE COMPLETED WITH SE API ACCOUNT

        ########################
        # SET WHICH FACE API TO USE:
        ########################
        self.faceAPI = api  # or "FacePlusePlus" or "IBM" or "Google" or "Amazon" or "SE"

        self.save_dir = save_dir+'\\images'
        self.raw_results = save_dir+'\\raw_results.txt'

        f = open(self.raw_results, 'a')
        f.write("ImageLocation,Race_Int,Gender_Int,Race,Gender,Face_Detected,Gender_Prediction,Gender_Correct\n")
        f.close()

        self.image_count=0

    ########################
    # GENERATE THE FACE IMAGE AND SEND TO THE FACE DETECTION:
    # Input = X <- a vector of parameters to configure the face.
    ########################
    def UserFun(self, x):

        # For RAND Exps:
        #regionLabel = numpy.floor(x[0])
        #genderLabel = x[1]

        # For BAYES Exps:
        regionLabel = numpy.argmax([x[0], x[1], x[2], x[3]])
        genderLabel = numpy.argmax([x[4], x[5]])

        test_num = 1
        test_label = str(int(regionLabel))
        test_gender = str(int(genderLabel))

        imageLocation = util.generate_fake_images(self.Gs, self.D, self.random_state, self.save_dir, test_label, test_gender, 1)   
        
        if imageLocation==None:
            return -1, -1, imageLocation
        else:
            timestr = datetime.utcnow().strftime('%Y%m%d%H%M%S%f') 
            filename_append = 'img_' + str(self.image_count) + '_cv_mode_' + timestr  +'.jpg'
            filename = '' + filename_append


            self.block_blob_service.create_blob_from_path(self.container_name, filename_append, imageLocation)#imageFolder+'temp.png')
            self.image_count = self.image_count+1

            gender=''
            if genderLabel == 0:
                gender = 'male'
            elif genderLabel == 1:
                gender = 'female'
            
            gt_region=''
            if regionLabel==0: #Race 0,1,2,3 = Black/Africa, white/Caucasian, AsianNE, AsianS
                gt_region = 'African'
            elif regionLabel==1:
                gt_region = 'Caucasian'
            elif regionLabel==2:
                gt_region = 'AsianNE'
            elif regionLabel==3:
                gt_region = 'AsianS'
            
            success, faces, faceDetected, genderPrediction = self.computeFaceDetection(filename, self.faceAPI, gender)
            FDsuccess = faceDetected
            GDsuccess = success

            if success==None:
                os.remove(imageLocation)
                return -1, -1, imageLocation
            else:
                print(imageLocation+" "+str(test_label)+" "+str(test_gender)+" "+gt_region[0:6]+" "+gender[0:3]+" "+str(faceDetected)+" "+str(genderPrediction)+" "+str(success))
                f = open(self.raw_results, 'a')
                f.write(imageLocation+","+str(test_label)+","+str(test_gender)+","+gt_region+","+gender+","+str(faceDetected)+","+str(genderPrediction)+","+str(success)+'\n')
                f.close()
                return FDsuccess, GDsuccess, imageLocation 
            

    ########################
    # METHODS TO RUN THE DIFFERENT FACE APIs:
    ########################

    ## For calling the Microsoft API. Provide a face image URL, file name and the gender label as reference: 
    def computeFaceDetectionMicrosoft(self,face_image_url, file_name, gender):

        table_name = 'Microsoft'
        partition_name = 'Microsoft'

        data = {'url': face_image_url}
        response = requests.post(self.msft_face_detection_url, params=self.msft_params, headers=self.msft_headers, json=data)              
        faces = response.json()  

        success = False
        genderPrediction = 'None'
        faceDetected = False
        if len(faces) == 0:
            success = False
            faceDetected = False
        elif len(faces)>0:
            faceDetected = True
            genderPrediction = faces[0]["faceAttributes"]["gender"]
            if gender==genderPrediction:
                success = True

        face_entry = Entity()
        face_entry.PartitionKey = partition_name
        face_entry.RowKey = file_name
        face_entry.Result = json.dumps(faces)
        face_entry.DetectionSuccess = success
        
        self.table_service.insert_entity(table_name, face_entry)
        
        return success, faces, faceDetected, genderPrediction.lower()

    ## For calling the SightEngine API. Provide a face image URL, file name and the gender label as reference: 
    def computeFaceDetectionSightEngine(self,face_image_url, file_name, gender):

        table_name = 'SightEngine'
        partition_name = 'SightEngine'

        output = self.SEclient.check('face-attributes').set_url(face_image_url)
           
        r = response.json()  
        faces = r["faces"]

        success = False
        genderPrediction = 'None'
        faceDetected = False
        if len(faces) == 0:
            success = False
            faceDetected = False
        elif len(faces)>0:
            faceDetected = True
            femaleProb = faces[0]["attributes"]["female"]
            maleProb = faces[0]["attributes"]["male"]
            if femaleProb>maleProb:
                genderPrediction = "female"
            else:
                genderPrediction = "male"
            if gender==genderPrediction:
                success = True

        face_entry = Entity()
        face_entry.PartitionKey = partition_name
        face_entry.RowKey = file_name
        face_entry.Result = json.dumps(faces)
        face_entry.DetectionSuccess = success
        
        self.table_service.insert_entity(table_name, face_entry)
        
        return success, faces, faceDetected, genderPrediction.lower()

    ## For calling the Face++ API. Provide a face image URL, file name and the gender label as reference: 
    def computeFaceDetectionFacePlusPlus(self,face_image_url, file_name, gender):

        table_name = 'FacePlusPlus'
        partition_name = 'FacePlusPlus'
        
        boundary = '----------%s' % hex(int(time.time() * 1000))

        data = []
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_key')
        data.append(self.faceplusplus_key)
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'api_secret')
        data.append(self.faceplusplus_secret)
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'return_attributes')
        data.append('gender')        
        data.append('--%s' % boundary)
        data.append('Content-Disposition: form-data; name="%s"\r\n' % 'image_url')
        data.append(face_image_url)        
        data.append('--%s--\r\n' % boundary)
        
        http_body='\r\n'.join(data)
        req=urllib.request.Request(self.faceplusplus_http_url)
        req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
        req.data = str.encode(http_body)
        try:
            resp = urllib.request.urlopen(req, timeout=5)    
            
            qrcont=resp.read().decode("utf-8")
            faces = json.loads(qrcont)

            success = False
            faceDetected = False
            genderPrediction = 'None'
            if 'faces' in faces.keys(): 
                faceDetected = True
                genderPrediction = faces["faces"][0]["attributes"]["gender"]["value"]
                if gender.lower()==genderPrediction.lower():
                    success = True
            else:
                success = None
                faceDetected = None

            time.sleep(2)
            face_entry = Entity()
            face_entry.PartitionKey = partition_name
            face_entry.RowKey = file_name
            face_entry.Result = json.dumps(faces)
            face_entry.DetectionSuccess = success
        
            self.table_service.insert_entity(table_name, face_entry)
            
            return success, faces, faceDetected, genderPrediction.lower()
            
        except urllib.request.HTTPError as e:
            return None, None, None, None

    ## For calling the IBM API. Provide a face image URL, file name and the gender label as reference: 
    def computeFaceDetectionIBM(self,face_image_url, file_name, gender):

        table_name = 'IBM'
        partition_name = 'IBM'
        
        urllib.request.urlretrieve(face_image_url, file_name)        
        with open(file_name, 'rb') as image_file:
            response = self.IBM_visual_recognition.detect_faces(image_file)       
    
        faces = response['images'][0]['faces']

        success = False
        faceDetected = False
        genderPrediction = 'None'
        if len(faces) == 0:
            success = False
            faceDetected = False
        elif len(faces)>0:
            faceDetected = True
            genderPrediction = faces[0]["gender"]["gender"]
            if gender.lower()==genderPrediction.lower():
                success = True
    
        face_entry = Entity()
        face_entry.PartitionKey = partition_name
        face_entry.RowKey = file_name
        face_entry.Result = json.dumps(response)
        face_entry.DetectionSuccess = success
        
        self.table_service.insert_entity(table_name, face_entry)    
        
        return success, faces, faceDetected, genderPrediction.lower()

    ## For calling the Google API. Provide a face image URL, file name and the gender label as reference: 
    def computeFaceDetectionGoogle(self,face_image_url, file_name, gender):

        table_name = 'Google'
        partition_name = 'Google'

        self.clientImgAnnotator = vision.ImageAnnotatorClient()
        image = vision.types.Image()
        image.source.image_uri = face_image_url

        try:
            response = self.clientImgAnnotator.face_detection(image=image)
            r = MessageToDict(response, preserving_proto_field_name = True)
            
            success = False
            faceDetected = False
            genderPrediction = 'None'
            if len(r) == 0:
                success = False
                faceDetected = False
                faces=[]
            elif len(r)>0:
                faceDetected = True
                success = False
                faces = r['face_annotations']
    
                face_entry = Entity()
                face_entry.PartitionKey = partition_name
                face_entry.RowKey = file_name
                face_entry.Result = json.dumps(faces)
                face_entry.DetectionSuccess = success
        
                self.table_service.insert_entity(table_name, face_entry) 
            
            return success, faces, faceDetected, genderPrediction.lower()

        except WatsonApiException as ex:
            print("Method failed with status code " + str(ex.code) + ": " + ex.message)

    ## For calling the Amazon API. Provide a face image URL, file name and the gender label as reference: 
    def computeFaceDetectionAmazon(self,face_image_url, file_name, gender):
        
        table_name = 'FFS3'
        partition_name = 'Amazon'
        
        urllib.request.urlretrieve(face_image_url, file_name)
        with open(file_name, 'rb') as image:
            response = self.amazon_client.detect_faces(Image={'Bytes': image.read()}, Attributes=['ALL'])
        
        faces = response['FaceDetails']
        
        success = False
        faceDetected = False
        genderPrediction = 'None'
        if len(faces) == 0:
            success = False
            faceDetected = False
        else:
            faceDetected = True
            genderPrediction = faces[0]["Gender"]["Value"]
            if gender.lower()==genderPrediction.lower():
                success = True
            else:
                success = False
        
        face_entry = Entity()
        face_entry.PartitionKey = partition_name
        face_entry.RowKey = file_name
        face_entry.Result = json.dumps(faces)
        face_entry.DetectionSuccess = success
        self.table_service.insert_entity(table_name, face_entry)
        
        return success, faces, faceDetected, genderPrediction.lower()


    def computeFaceDetection(self,face_image_url, service, gender):
        
        file_name = face_image_url.split('/')[-1]
        
        if service == 'Microsoft':
            return self.computeFaceDetectionMicrosoft(face_image_url, file_name, gender)        
        elif service == 'FacePlusPlus':
            return self.computeFaceDetectionFacePlusPlus(face_image_url, file_name, gender)         
        elif service == 'IBM':
            return self.computeFaceDetectionIBM(face_image_url, file_name, gender)         
        elif service == 'Google':
            return self.computeFaceDetectionGoogle(face_image_url, file_name, gender)
        elif service == 'Amazon':
            return self.computeFaceDetectionAmazon(face_image_url, file_name, gender)
        elif service == 'SightEngine':
            return self.computeFaceDetectionAmazon(face_image_url, file_name, gender)                    
        
        else:
            raise ValueError(service + ' is not a valid service name')
