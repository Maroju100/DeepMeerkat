from __future__ import absolute_import
import os
import random
import glob
import csv
import tempfile
import sys
if sys.version_info >= (3, 0):
    from urllib import urlparse
else:
    from urlparse import urlparse
import argparse

class Organizer:
    def __init__(self):
        
        
        #Training
        self.train_positives_files=glob.glob("Samples/Training/Positives/*.jpg")       
        print( "Positive training samples: %d" %(len( self.train_positives_files)))  
        
        #negatives
        self.train_negatives_files=glob.glob("Samples/Training/Negatives/*.jpg")       
        print( "Negative training samples: %d" %(len( self.train_negatives_files)))     
                    
        #shuffle negatives and take a sample equal to the size of the positives
        random.shuffle(self.train_negatives_files)
        
        #cut the file to match positives
        self.train_negatives_files=self.train_negatives_files[:len(self.train_positives_files)]
        
        print( "Negative Training Samples: %d" % (len( self.train_negatives_files)))          
        
        ##Testing
        
        #testing
        self.test_positives_files=glob.glob("Samples/testing/Positives/*.jpg")       
        print( "Positive testing samples: %d" %(len( self.test_positives_files)))  
    
        #negatives
        self.test_negatives_files=glob.glob("Samples/testing/Negatives/*.jpg")       
        print( "Negative testing samples: %d" %(len( self.test_negatives_files)))     
    
        #shuffle negatives and take a sample equal to the size of the positives
        random.shuffle(self.test_negatives_files)
    
        #cut the file to match positives
        self.test_negatives_files=self.test_negatives_files[:len(self.test_positives_files)]
    
        print( "Negative testing Samples: %d" % (len( self.test_negatives_files)))        
        
    def write_data(self):
        
        ##Training
        
        with open("Samples/Training/trainingdata.csv","wb") as f:
            writer=csv.writer(f)
            for eachrow in  self.train_positives_files:
                writer.writerow([str(eachrow),"positive"])
            for eachrow in  self.train_negatives_files:
                writer.writerow([str(eachrow),"negative"])
        
        ##Testing
        with open("Samples/Testing/testingdata.csv","wb") as f:
            writer=csv.writer(f)
            for eachrow in  self.test_positives_files:
                writer.writerow([str(eachrow),"positive"])
            for eachrow in  self.test_negatives_files:
                writer.writerow([str(eachrow),"negative"])

        #write dict file 
        with open("Samples/dict.txt","wb") as f:
            f.write("positive"+"\n")
            f.write("negative")
            f.close()
        
if __name__ == "__main__":
    print(__name__)
    p=Organizer()
    p.write_data()