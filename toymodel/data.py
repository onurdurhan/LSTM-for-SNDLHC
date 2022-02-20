import torch
from torch.utils.data import IterableDataset, DataLoader
import ROOT, sys, os, getopt
import rootUtils as ut
from rootpyPickler import Unpickler
from decorators import *
import operator
import argparse
import math
from array import array
from torch.nn.utils.rnn import pack_sequence
import random


parser=argparse.ArgumentParser()
parser.add_argument("-f","--files",dest="inputFile",help="no help",required=True)
parser.add_argument("-g", "--geoFile", dest="geoFile", help="geofile", required=True)
options=parser.parse_args()
inputFile=options.inputFile
PDG=ROOT.TDatabasePDG.Instance()
f=ROOT.TFile(inputFile)
trans2local = False

fgeo = ROOT.TFile.Open(options.geoFile)
from ShipGeoConfig import ConfigRegistry
from rootpyPickler import Unpickler
#load geo dictionary
upkl    = Unpickler(fgeo)
snd_geo = upkl.load('ShipGeo')

# -----Create geometry----------------------------------------------
import shipLHC_conf as sndDet_conf

run = ROOT.FairRunSim()
modules = sndDet_conf.configure(run,snd_geo)
sGeo = fgeo.FAIRGeom
modules['Scifi'].SiPMmapping()
lsOfGlobals = ROOT.gROOT.GetListOfGlobals()
lsOfGlobals.Add(modules['Scifi'])
lsOfGlobals.Add(modules['MuFilter'])
nav = ROOT.gGeoManager.GetCurrentNavigator()

sTree=f.cbmsim
A,B = ROOT.TVector3(),ROOT.TVector3() 

class HitDataset(IterableDataset):


    def __init__(self,train_test):
        self.label2id = {"anything_else":0,"muon":1}
        self.train_test = train_test
        # sample data
        if True :
            self.data=[]
            nStart = 0
            nEnd = sTree.GetEntries()
            print("TEST DATA FOR",nEnd, " EVENTS" )
            for event in range(nStart,nEnd):
                sTree.GetEvent(event)
                if sTree.MCTrack[0].GetP() > 500. : continue
                t1,t2,t3,t4,t5 =0,0,0,0,0
                for hit in sTree.Digi_MuFilterHits:
                    detID = hit.GetDetectorID()
                    s = detID//10000
                    l  = (detID%10000)//1000  # plane number
                    bar = (detID%1000)
                    if hit.GetName()  == 'MuFilterHit':
                        system = hit.GetSystem()
                        modules['MuFilter'].GetPosition(detID,A,B)
                        globA,locA = array('d',[A[0],A[1],A[2]]),array('d',[A[0],A[1],A[2]])
                    if trans2local:   nav.MasterToLocal(globA,locA)
                    Z = A[2]
                    if hit.isVertical():
                        X = locA[0]
                    else:                         
                        Y = locA[1]
#                    print("s, l, bar ",s, l, bar, Y)
                    if not hit.GetSystem() == 2: continue
                    layer=(detID-2E4)//1E3
                    if layer == 0:
                        t1+=1
                    if layer == 1:
                        t2+=1
                    if layer == 2:
                        t3+=1
                    if layer == 3:
                        t4+=1
                    if layer == 4:
                        t5+=1
                       
                t0 = sTree.Digi_ScifiHits.GetEntries()
#                if sTree.MCTrack[0].GetP()>500. : continue
                if t0==0 and t1==0 and t2 == 0 and t3 == 0 and t4 == 0 and t5 == 0  :continue
                hitlist=[t0,t1,t2,t3,t4,t5]
#                norm = [float(i)/max(hitlist) for i in hitlist]
                temp = hitlist, sTree.MCTrack[0].GetP()
                random.shuffle(self.data)
                self.data.append(temp)
            random.shuffle(self.data)
            random.shuffle(self.data)
            random.shuffle(self.data)
            border = (90*len(self.data)/100)
            border = int(border)
            random.shuffle(self.data)
            if train_test == "train":
                self.data = self.data[:border]
            if train_test == "test":
                self.data = self.data[border:]
#         
    def __iter__(self):
        for x in self.data:
            yield torch.tensor(x[0]).float(), torch.tensor(x[1]).float()
    def __len__(self):
        return len(self.data)
