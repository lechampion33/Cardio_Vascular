import pandas as pd
from scipy.io import loadmat
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt


class ECG:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = loadmat(self.file_path)
        self.Leads = self.data['ECG'][0][0][0]

    def getImage(self, image_path):
        """
        this functions gets user image
        return: user image
        """
        image = imread(image_path)
        return image

    def DividingLeads(self):
        """
        this functions divide leads to two groups
        return: two groups of leads
        """
        Leads_1_12 = self.Leads[0:12, :]
        Long_Lead_13 = self.Leads[12:13, :]
        fig, axs = plt.subplots(12, 1, figsize=(10, 20))
        for i in range(12):
            axs[i].plot(Leads_1_12[i, :])
            axs[i].set_title(f'Lead {i+1}')
        fig.savefig('static/Leads_1-12_figure.png')
        fig1, axs1 = plt.subplots(1, 1, figsize=(10, 5))
        axs1.plot(Long_Lead_13[0, :])
        axs1.set_title('Lead 13')
        fig1.savefig('static/Long_Lead_13_figure.png')
        return Leads_1_12, Long_Lead_13

    def PreprocessingLeads(self, Leads_1_12, Long_Lead_13):
        """
        this functions preprocess leads
        return: preprocessed leads
        """
        b, a = butter(3, 0.05, 'low')
        Leads_1_12_filtered = filtfilt(b, a, Leads_1_12, axis=1)
        Long_Lead_13_filtered = filtfilt(b, a, Long_Lead_13, axis=1)
        fig2, axs2 = plt.subplots(12, 1, figsize=(10, 20))
        for i in range(12):
            axs2[i].plot(Leads_1_12_filtered[i, :])
            axs2[i].set_title(f'Lead {i+1}')
        fig2.savefig('static/Preprossed_Leads_1-12_figure.png')
        fig3, axs3 = plt.subplots(1, 1, figsize=(10, 5))
        axs3.plot(Long_Lead_13_filtered[0, :])
        axs3.set_title('Lead 13')
        fig3.savefig('static/Preprossed_Leads_13_figure.png')
        return Leads_1_12_filtered, Long_Lead_13_filtered

    def SignalExtraction_Scaling(self, Leads_1_12_filtered):
        """
        this functions extract signal and scaling
        return: scaled signal
        """
        Leads_1_12_filtered = Leads_1_12_filtered
        fig4, axs4 = plt.subplots(12, 1, figsize=(10, 20))
        for i in range(12):
            axs4[i].plot(Leads_1_12_filtered[i, :])
            axs4[i].set_title(f'Lead {i+1}')
        fig4.savefig('static/Contour_Leads_1-12_figure.png')
        return Leads_1_12_filtered

    def CombineConvert1Dsignal(self, Leads_1_12_filtered):
        """
        this functions convert 2D signal to 1D signal
        return: 1D signal
        """
        Scaled_1DLead_1 = Leads_1_12_filtered.reshape(1, -1)
        df = pd.DataFrame(Scaled_1DLead_1)
        df.to_csv('static/Scaled_1DLead_1.csv', index=False)
        test_final = pd.read_csv('static/Scaled_1DLead_1.csv')
        location = 'static'
        return test_final, location



