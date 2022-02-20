import torch
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import ROOT
import rootUtils as ut
from model import MyLSTM
from data import HitDataset
from array import array
import numpy as np

ROOT.gROOT.SetBatch(ROOT.kTRUE)
#ROOT.gStyle.SetBarWidth(50)


hist={}
ut.bookHist(hist,"E_true","True",100,0.,1010.)
ut.bookHist(hist,"E_reco","Reco",100,0.,1010.)
ut.bookHist(hist,"reco_vs_true","reco_vs_true",100,0.,500.,100,0.,500.)
ut.bookHist(hist,"difference","diff",100,-1.,1.)
ut.bookHist(hist,"resolution","energy resolution",100,0.,1010.,100,-4.,4.)
ut.bookCanvas(hist,"energy_res","energy_res",700,500)
ut.bookCanvas(hist,"loss","loss",700,500)


def train(model,num_epochs = 300, init_lr=0.000001):
    train_data = HitDataset(train_test="train")
    train_iter = DataLoader(train_data, batch_size=1)
    optimizer = optim.Adam(model.parameters(),init_lr, weight_decay= 0.0001)
#    optimizer = optim.SGD(model.parameters(), init_lr)#, momentum = 0.05)
    mse_loss = nn.MSELoss()
#    mse_loss = nn.L1Loss()
#    loss_hist = []
#    gr1 = ROOT.TGraph()
    X = array('d',[])
    Y = array('d',[])
#    np=0
    for epoch in range(1, num_epochs+1):
        epoch_loss = 0
        for x, y in train_iter:
            optimizer.zero_grad()
            pred = model(x)
            loss = mse_loss(pred.reshape(1),y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_iter)
#        gr1.SetPoint(np,epoch,epoch_loss)
        X.append(epoch)
        Y.append(epoch_loss)
#        np+=1
        print(epoch_loss,"at epoch ",epoch)
        if epoch % 100 == 0 or epoch == 1:
            print('Epoch: [%d/%d]\tLoss : %.4f\t' % (epoch, num_epochs, epoch_loss))
    print("Training done") 
    return X,Y

def test(model):
    test_data = HitDataset(train_test="test")
    test_iter = DataLoader(test_data, batch_size=1)    
    model.eval()
#    gr2 = ROOT.TGraph()
#    gr2.SetFillColor(40)
#    gr2.SetMarkerStyle(20)
#    np=0
    X = array('d',[])
    Y = array('d',[])
    for x, y in test_iter:
        pred = model(x)
        hist["E_reco"].Fill(pred)
        hist["E_true"].Fill(y)
        hist["reco_vs_true"].Fill(pred,y)
        deltaE=pred-y
        hist["difference"].Fill(deltaE/y)
        hist["resolution"].Fill(y,deltaE/y)
#        gr2.SetPoint(np,y,deltaE/y)
        X.append(y)
        Y.append(deltaE/y)
#        np+=1
    return X,Y


def main():
    model = MyLSTM()
    print("Training...")
    hist["loss"].cd()
    rc = train(model, num_epochs=100, init_lr=0.00006)
    gr1 = ROOT.TGraph(len(rc[0]),rc[0],rc[1])
    gr1.Draw("AL")
    print("Test:")
    hist["energy_res"].cd()
    rc=test(model)
    gr2 = ROOT.TGraph(len(rc[0]),rc[0],rc[1])
    gr2.SetMarkerSize(1)
    gr2.SetFillColor(38)
    ROOT.gStyle.SetBarWidth(float(ROOT.gStyle.GetBarWidth()*0.5*200))
    hist["energy_res"].UseCurrentStyle()
    gr2.Draw("AP")
    hist["energy_res"].Update()
#    hist["energy_res"].Modified()
    f = ROOT.TFile("LSTM_result_5.root","RECREATE")
    f.cd()
    for key in hist:
        hist[key].Write()
    f.Close()
#    ROOT.gROOT.SetBatch(ROOT.kTRUE)
   

#    plt.plot(loss)
#    plt.xlabel('Epoch')
#    plt.ylabel('Loss')
#    plt.savefig("loss_curve.png")


if __name__ == "__main__":
#    ROOT.gStyle.SetBarWidth(10)
    rc=main()
