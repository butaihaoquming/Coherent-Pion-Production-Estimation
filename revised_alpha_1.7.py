import ROOT
import numpy as np
import scipy
from iminuit import Minuit
from pathlib import Path
from datetime import datetime
import array
import math
import inspect
import pandas as pd


file = ROOT.TFile.Open("/Users/jerry/Desktop/root/Root file/latest_data/Another New Upload Folder for Mehreen 10 March-selected/texp_CVOnly_MC.root")
file1 = ROOT.TFile.Open("/Users/jerry/Desktop/root/Root file/latest_data/Another New Upload Folder for Mehreen 10 March-selected/texp_CVOnly_Data.root")
chi_square_values = []
# This is the bins that will be used to extract data
custom_bins = np.array([0.000, 0.01, 0.020, 0.030, 0.045, 0.06, 0.080, 0.10, 0.125, 0.15, 
                        0.175, 0.20, 0.25, 0.30, 0.35, 0.4, 0.45, 0.50, 0.60, 0.80, 1.0])
custom_bins_list = custom_bins.tolist()
custom_bins_array = array.array('d', custom_bins_list)
template_hist = ROOT.TH1D("hist", "Filled Histogram", len(custom_bins_array)-1, custom_bins_array)
#So we can run a for loop to go over all the histograms
histogram_name =  ["A","B","C","D"]

#This is the eight histograms we will use
   
signal_name = "bin_Tpibins_picat_COH_Pions" 
background1_name = "bin_Tpibins_picat_1pip_w_LProton" 
background2_name = "bin_Tpibins_picat_1pip_w_LNeutron" 
data_hist_name = "bin_Tpibins_data"

extra_bkg1 = "bin_Tpibins_by_BKG_Label_QELikeBkg" 
extra_bkg2 = "bin_Tpibins_by_BKG_Label_KaonsOnly" 
extra_bkg3 = "bin_Tpibins_picat_NPi_and_Mesons_RES" 
extra_bkg4 = "bin_Tpibins_by_BKG_Label_Other" 
extra_bkg5 = "bin_Tpibins_picat_Npi_Other" 
extra_bkg6 = "bin_Tpibins_by_BKG_Label_Pi0Bkg" 
extra_bkg7 = "bin_Tpibins_by_BKG_Label_Wrong_Sign"  
extra_bkg8 = "bin_Tpibins_by_BKG_Label_NCbkg" 
extra_bkg9 = "bin_Tpibins_by_BKG_Label_FSBkg" 
extra_bkg10 = "bin_Tpibins_picat_Npip_Only" 
extra_bkg11 = "bin_Tpibins_picat_NPi_and_Mesons_DIS" 

migration_matrix = "Tpi_mig_COH_Pions"
i_labels = [f"i={i}" for i in range(6)]
m_labels = [f"m={m}" for m in range(6)]
k_labels = [f"k={k}" for k in range(20)]

tune_hist = [signal_name, background1_name, background2_name]
bkg_hist_list = [extra_bkg1, extra_bkg2, extra_bkg3, extra_bkg4, extra_bkg5, extra_bkg6,\
             extra_bkg7, extra_bkg8, extra_bkg9, extra_bkg10, extra_bkg11]

#Now the parameters are all listed as follows: we have 18 numbers here, the corresponds to: the first three are
#beta_1, gamma_1,c_1, and the second set of three numbers are beta_2, gamma_2,c_2 etc. The phi values are stored differently
#in the next list, or we can combined the two lists together.               
params = [1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1]

phi_values = []
"""[1.416884788583217, 0.9719475260085099, -0.2046560516656213, 1.2425771010558313, 1.0041249706013857, \
 -0.1945659413348425, 1.0899354758234598, 1.0180619685543417, -0.1407768638867592, 0.9545522804106907, \
    0.9888587530839709, -0.05236944327717352, 0.9224763495733007, 0.9931619184102711, -0.015083978302926872, \
        0.9266515114250399, 0.9209116198857402, -0.0008889373789207735, 0.8974468623933699, 0.7267503021861148, \
            0.8036360097473975, 1.0469640746156468, 0.3600703302817001, 0.8044756774405661]"""


#[1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1]
alpha_list = np.zeros((6,4))
migration_matrix_1 = np.zeros((6, 6, 4))
pre_process_migration = np.zeros((8, 8, 4))
migration_matrix_2 = np.zeros((6, 6, 4))
Coh_matrix = np.zeros((6, 20, 4))
true_Tpion = np.zeros((6, 20, 4))
#create the results dictionary without any keys
results = {}
other_i = 1
runs = 0
hist_senario = ["0 < T_{#pi^{+}} < 30 MeV", "30 < T_{#pi^{+}} < 60 MeV", \
                "60 < T_{#pi^{+}} < 100 MeV", "100 < T_{#pi^{+}} < 150 MeV", \
                "150 < T_{#pi^{+}} < 200 MeV", "200 < T_{#pi^{+}} < 250 MeV"]
Eava = ["0 < Available Energy - T_{#pi^{+}} < 25 MeV", "25 < Available Energy - T_{#pi^{+}} < 50 MeV", \
        "50 < Available Energy - T_{#pi^{+}} < 75 MeV", "75 < Available Energy - T_{#pi^{+}} < 150 MeV"]
plot_name = ["Unscaled1.png", "Scaled1.png", "Unscaled2.png", "Scaled2.png", "Unscaled3.png", "Scaled3.png", \
              "Unscaled4.png", "Scaled4.png"]

#This will return the data set in a list and we can process the data based on that
def _chi2_wrapper(*phis):
    # phis is a tuple of floats
    return calculate_chi_square(list(phis))

def chi2_barrier(*phis):
    # phis is a tuple of length 24
    # blunt barrier: if any of the *last six* φ’s is negative, give a huge χ²
    if any(phi < 0 for phi in phis[-6:]):
        return 1e300
    return calculate_chi_square(list(phis))

def get_data(data_hist):
    data_list = []
    for i in range(1, data_hist.GetNbinsX() + 1):
        bin_content = data_hist.GetBinContent(i)
        data_list.append(bin_content)
    return data_list

#This will return the chi-square value of the histograms.
def chi_square(prediction, data):
    total_chi_square_value = 0.0
    for j in range(0,4):
        for i in range(0,6):
            for k in range(0,20):
                if data[i,k,j] != 0:
                    chi_square = (prediction[i,k,j] - data[i, k, j])**2 / data[i, k, j]
                    total_chi_square_value = total_chi_square_value + chi_square
                else:
                    pass
    return total_chi_square_value

#By inputing the four histograms' values, we can calculate the prediction value for each bin.
def get_prediction(results, parameters, TH1D_1, TH1D_2):
    #Since we need to go over a big loop and generate the whole prediction values for all situations, we need to generate a matrix so we can 
    #get access to the data very easily.
    rows =6
    columns = 3
    params_matrix = [parameters[i* columns: (i+1) * columns] for i in range(rows)]
    #Now we get a 6*3 matrix which each row represent the i range and each row contains beta, gamma can exp constant c.
    #Next, we extract the required datas and try to run the prediction fucntion to generate a 6*20 matrix and store all
    #the prediction values in it.

    #First generate an 6*20 0 matrix.
    prediction_matrix = np.zeros((6, 20, 4))
    data_matrix = np.zeros((6, 20, 4))
    #Then we create the for loop to run through 6 i bins and 4 j bins.

    for i in range(1,7):
        for j in range(0,4):
            #Now we are extrating the datas from the nested dictionary.
            Coh_list  = results[i][j]["Coh"]
            #We are calling the "Coh" term stored in the section [1][0], for each section I previous stored
            #"Coh" "bkg1" "bkg2" "combined_bkg" "data". for each nested loop we have two index, so we can use the 
            #indices to acess a specific section and hence use the names as keys to get the correct list.
            bkg1_list = results[i][j]["bkg1"]
            bkg2_list = results[i][j]["bkg2"]
            combined_bkg_list = results[i][j]["combined_bkg"]
            data_list = results[i][j]["data"]
            #print("Coh+list:", Coh_list, bkg1_list)
            #Next step is to revise the value of the bkg1 and bkg2 with the exponential constants. After we do that we are able to input all
            #the ingredients into the prediction calculation.
            exp_bkg1_list = exponential(TH1D_1, bkg1_list,params_matrix[i-1][2])
            exp_bkg2_list = exponential(TH1D_2, bkg2_list, params_matrix[i-1][2])

            #Next, we use another for loop to go through the each 20 terms in the lists to calcualte the prediction, at the same time
            # we need to extract the constants from the previously generated parameter matrix.
            for k in range(len(bkg1_list)):
                prediction_value = Coh_list[k] + params_matrix[i-1][0] * exp_bkg1_list[k] + params_matrix[i-1][1] * exp_bkg2_list[k] + combined_bkg_list[k]
                #Now assign the value to the correct position in matrix.
                prediction_matrix[i-1, k, j] = prediction_value
                data_matrix[i-1, k, j] = data_list[k]
    return prediction_matrix, data_matrix
  
#This function rebin the histograms acoording to the bin we assigned and normalize the bin value.
def rebin_and_normalize(hist, name):
    rebinned_hist = hist.Rebin(len(custom_bins)-1, name, custom_bins)
    for bin_idx in range(1, rebinned_hist.GetNbinsX() + 1):
        bin_content = rebinned_hist.GetBinContent(bin_idx)
        bin_width = rebinned_hist.GetBinWidth(bin_idx)
        # Divide the bin content by the bin width
        if bin_width > 0:
            rebinned_hist.SetBinContent(bin_idx, bin_content / bin_width)
    return rebinned_hist

def normalization(hist1,name):
    norm_name = f"{name}_normalized"
    hist2 = hist1.Clone(norm_name)
    for bin_idx in range(1, hist1.GetNbinsX() + 1):
        bin_content = hist1.GetBinContent(bin_idx)
        bin_binwidth = hist1.GetBinWidth(bin_idx)
        hist2.SetBinContent(bin_idx, (bin_content / bin_binwidth))
    return hist2

def normalization_list(list):
    bin_width = []
    revised_list = []
    for i in range(0,20):
        bin_width.append(custom_bins[i+1] - custom_bins[i])
    for list_element in range(0,20):
        revised_list.append(list[list_element] / bin_width[list_element])
    return revised_list

def rebin_hist(hist, name):
    rebinned_hist = hist.Rebin(len(custom_bins)-1, name, custom_bins)
    for bin_idx in range(1, rebinned_hist.GetNbinsX() + 1):
        bin_content = rebinned_hist.GetBinContent(bin_idx)
        rebinned_hist.SetBinContent(bin_idx, bin_content)
    return rebinned_hist

def create_TH1D(list_value, name):
    h = ROOT.TH1D(name, name, len(custom_bins_array)-1, custom_bins_array)
    for bin_idx in range(1, h.GetNbinsX() + 1):
        bin_content = list_value[bin_idx - 1]
        h.SetBinContent(bin_idx, bin_content)
    return h
    
#This function rescale the bin value according to a constant. Now its 1. Rescaling is no longer needed.
def rescaling_histogram(hist1):
    for bin_idx in range(1, hist1.GetNbinsX() + 1):
            bin_content = hist1.GetBinContent(bin_idx)
            bin_value = bin_content / 3
            hist1.SetBinContent(bin_idx, bin_value)
    return hist1

def rescaling_histogram_1(hist1,c):
    #print("c is:", c)
    for bin_idx in range(1, hist1.GetNbinsX() + 1):
            bin_content = hist1.GetBinContent(bin_idx)
            #print("bin_content:", bin_content)
            bin_value = bin_content * c
            #print("bin_value:", bin_value)
            hist1.SetBinContent(bin_idx, bin_value)
    return hist1

def rescaling_histogram_alpha(hist1, value):
    for bin_idx in range(1, hist1.GetNbinsX() + 1):
            bin_content = hist1.GetBinContent(bin_idx)
            hist1.SetBinContent(bin_idx, (bin_content * value))
    return hist1

def rescaling_histogram_beta(hist1, value):
    for bin_idx in range(1, hist1.GetNbinsX() + 1):
            bin_content = hist1.GetBinContent(bin_idx)
            hist1.SetBinContent(bin_idx, (bin_content * value))
    return hist1

def rescaling_histogram_gamma(hist1, value):
    for bin_idx in range(1, hist1.GetNbinsX() + 1):
            bin_content = hist1.GetBinContent(bin_idx)
            hist1.SetBinContent(bin_idx, (bin_content * value))
    return hist1

def exponential(hist, list, c):
    revised_list = []
    for i in range(len(list)):
        binwidth = hist.GetBinCenter(i+1)
        revised_list.append(list[i] * np.exp(c * binwidth))
    #print(revised_list)
    return revised_list

def hist_exp(hist, c):
    for i in range(1, hist.GetNbinsX() + 1):
        binwidth = hist.GetBinCenter(i)
        new_bin = hist.GetBinContent(i) * np.exp(c * binwidth)
        hist.SetBinContent(i, new_bin)

def show_y_value(hist):
    y_value = []
    for i in range(1, hist.GetNbinsX()+1):
        y_value.append(hist.GetBinContent(i))
    #print(y_value)
    return y_value

def show_binwidth(hist):
    y_value = []
    for i in range(1, hist.GetNbinsX() + 1):
        y_value.append(hist.GetBinWidth(i))
    #print(y_value)

def build_sheet_blocks_6x20(arrs_by_name, i_labels, k_labels):
    """
    Combine several 6x20x4 arrays into one sheet:
    columns: j, i, then groups [<name> x k=1..20] for each array name.
    Rows are stacked by j with a blank spacer row between j-slices.
    """
    # sanity check shapes & same depth
    Js = {arr.shape[2] for arr in arrs_by_name.values()}
    assert len(Js) == 1, f"All arrays must have same third-dimension depth; got depths {Js}"
    J = Js.pop()

    parts = []
    for j in range(J):
        # build per-matrix blocks with (name on top, k as subcolumns)
        blocks = []
        for name, arr in arrs_by_name.items():
            # arr[:, :, j] assumed (i,k) at this slice
            df = pd.DataFrame(arr[:, :, j], index=i_labels, columns=k_labels)
            df = df.reset_index().rename(columns={'index': 'i'})  # make row labels a column
            df.insert(0, 'j', f'j={j}')                            # put j first

            data = df.drop(columns=['j', 'i'])
            data.columns = pd.MultiIndex.from_product([[name], data.columns])
            df = pd.concat([df[['j', 'i']], data], axis=1)
            blocks.append(df)

        # merge on ['j','i'] so we keep one j/i pair at the left
        wide = blocks[0]
        for b in blocks[1:]:
            wide = wide.merge(b, on=['j', 'i'])

        parts.append(wide)

        # blank spacer row between slices (not after the last)
        if j != J - 1:
            spacer = pd.DataFrame([[np.nan] * wide.shape[1]], columns=wide.columns)
            parts.append(spacer)

    big = pd.concat(parts, axis=0, ignore_index=True)
    return big

def build_sheet_blocks_6x6(arrs_by_name, i_labels, m_labels):
    """
    Same idea but for 6x6x4 arrays: (i,m,j) -> columns are j, i, then groups [<name> x m=1..6].
    """
    Js = {arr.shape[2] for arr in arrs_by_name.values()}
    assert len(Js) == 1, f"All arrays must have same third-dimension depth; got depths {Js}"
    J = Js.pop()

    parts = []
    for j in range(J):
        blocks = []
        for name, arr in arrs_by_name.items():
            df = pd.DataFrame(arr[:, :, j], index=i_labels, columns=m_labels)
            df = df.reset_index().rename(columns={'index': 'i'})
            df.insert(0, 'j', f'j={j}')
            data = df.drop(columns=['j', 'i'])
            data.columns = pd.MultiIndex.from_product([[name], data.columns])
            df = pd.concat([df[['j', 'i']], data], axis=1)
            blocks.append(df)

        wide = blocks[0]
        for b in blocks[1:]:
            wide = wide.merge(b, on=['j', 'i'])

        parts.append(wide)
        if j != J - 1:
            spacer = pd.DataFrame([[np.nan] * wide.shape[1]], columns=wide.columns)
            parts.append(spacer)

    big = pd.concat(parts, axis=0, ignore_index=True)
    return big

def extract_migration():
    for avaE in range(0,4):
        for i_bin in range(1,7):
            y_bin_min = i_bin
            y_bin_max = i_bin
            migration_name = histogram_name[avaE] + migration_matrix
            migration_hist = file.Get(migration_name)
            project_migration =  migration_hist.ProjectionY("migration", y_bin_min, y_bin_max)
            migration_data = get_data(project_migration)
            global migration_matrix_2, pre_process_migration
            pre_process_migration[i_bin-1, :, avaE] = migration_data
        
        for i_bin_1 in range(1,7):
            column = pre_process_migration[:, i_bin_1-1, avaE]
            data_total_value = 0

            for data in column:
                data_total_value = data_total_value + data

            if data_total_value == 0:
                normalized_migration = [0,0,0,0,0,0,0,0]
            else: 
                normalized_migration = []
                for data in column: 
                    normalized_migration.append(data/data_total_value)
            normalized_migration = normalized_migration[:-2]
            migration_matrix_2[:, i_bin_1-1, avaE] = normalized_migration
    return


#So this is the main program we are running, everything is in this for loop because the histograms name only differ by a character, 
#ABCD, so we the body of the histograms, we just have to add the first character to make it a valid histogram name.
#And hence, it will run over the whole situaions and give us a chi-square value for a certain T_pion value.\
def calculate_chi_square(inputed_parameters):
    total_chi_square = 0
    global runs
    runs = runs +1
    print(runs)


    #since we need to run the whole list through everything, we need to store our data in the nested dictionary. Here we have two indexes,
    #just like we have a two layer for loop structure. Still we are trying to output the following data: for each hist slice and each i value we are generating
    # a list to cover the value of the different values which the prediction function requries to generate the prediction value and further input into chi-square value
    # minimizer to find the parameters to find the minimized chi-sqaure value.
    for hist_slice in range(1,7):
        #Here we write the required nested dictionary, here we generate the first layer of the dictonary which uses the "key" hist_slice

        results[hist_slice] = {}
        
        #The following is the subdictionary which we will assign the second "key" i to distinguish the Eavailable energy.
        for avaE in range(0,4):
            results[hist_slice][avaE] = {}
            #It extracts the last six terms of the param list, which is the six phi values.
            global phi_values
            phi_values = inputed_parameters[-6:]
            #Here we extract the correct slice from the histograms, which corresponds to the different Tpion energies.
            y_bin_min = hist_slice
            y_bin_max = hist_slice
            # So the following combines the first character with the main body of the histogram names
            data_hist_name1 = histogram_name[avaE] + data_hist_name
            tune_hist_names = []
            for names in tune_hist: 
                hist_name = histogram_name[avaE] + names
                tune_hist_names.append(hist_name)
            
            bkg_hist_names = []
            for names1 in bkg_hist_list:
                hist_name_1 = histogram_name[avaE] + names1
                bkg_hist_names.append(hist_name_1)

            
            
            #tune_hist_names contains the full name of tuned histograms, stored in the list
            #bkg_hist_names contains the full name of the extrabackground histograms, soted in the list.


            #print(combined_signal_name,combined_background1_name,combined_background2_name,combined_data_name,\
            # combined_extra_bkg1,combined_extra_bkg2,combined_extra_bkg3,combined_extra_bkg4)
        
            #And here we extract the histograms from the file
            #sicne data is special, not in the MC file but in the data file
            data_hist = file1.Get(data_hist_name1)
            tune_TH2D = []
            for hists in tune_hist_names:
                tune_TH2D_hist = file.Get(hists)
                tune_TH2D.append(tune_TH2D_hist)

            bkg_TH2D = []
            for hists1 in bkg_hist_names:
                bkg_TH2D_hist = file.Get(hists1)
                bkg_TH2D.append(bkg_TH2D_hist)
            #tune_TH2D contains the list of TH2D histograms.
            #bkg_TH2D contains the list of TH2D extra background histograms extrated from the files

        


            #now tune_TH1D contains the projection of TH2D histograms
            #bkg_TH1D constians the projection of the extrabackground TH2D histograms

            # Projecting tune_TH2D histograms
            tune_TH1D = []
            for idx, TH2D_hist in enumerate(tune_TH2D):
                projection_name = f"tuned_hist{idx}"  # You can customize the naming as needed
                TH1D_hist = TH2D_hist.ProjectionX(projection_name, y_bin_min, y_bin_max)
                tune_TH1D.append(TH1D_hist)

            # Projecting bkg_TH2D histograms
            bkg_TH1D = []
            for idx, bkg_TH2D_hist in enumerate(bkg_TH2D):
                projection_name = f"extra_proj_bkg{idx+1}"  # Ensures names like extra_proj_bkg1, etc.
                extra_proj_bkg = bkg_TH2D_hist.ProjectionX(projection_name, y_bin_min, y_bin_max)
                bkg_TH1D.append(extra_proj_bkg)

            # Projecting data_hist
            data_proj = data_hist.ProjectionX("data_proj", y_bin_min, y_bin_max)

            # Combining backgrounds
            combined_background = bkg_TH1D[0].Clone("combined_background")
            for bkg_hist in bkg_TH1D[1:]:
                combined_background.Add(bkg_hist)

            


            #Next we rebin the data.
            #now the tune_Th1D contains the TH1D histograms rebinned. Apart from that there are combined_background and data.
            for idx, rebins in enumerate(tune_TH1D):
                rebinning_name = f"rebinned_hist{idx}"
                rebinned_histogram = rebin_hist(rebins, rebinning_name)
                tune_TH1D[idx] = rebinned_histogram
            rebinned_combined_background = rebin_hist(combined_background, "rebinned_combined_background")
            rebinned_data = rebin_hist(data_proj, "rebinned_data")

            
        
            #the tune_TH1D contains the rescaled histograms. The data isnot rescaled.
            #Rscaling the histograms by 4
            #for idx1, rescale in enumerate(tune_TH1D):
            #   rescaled_histogram = rescaling_histogram(rescale)
            #    tune_TH1D[idx1] = rescaled_histogram
            #rebinned_combined_background = rescaling_histogram(rebinned_combined_background)

            #We here apply a constant exp(ct) to background 1 and 2.
            #We extract the values of the rebinned data in the histograms, and what we get is lists.
            


            #HERE we get the values of the data lists.
            signal_data_list = get_data(tune_TH1D[0])
            print(f"The raw signal data for i={hist_slice}, j={avaE} is:", signal_data_list)

            


            background1_data_list = get_data(tune_TH1D[1])
            background2_data_list = get_data(tune_TH1D[2])
            #print(background1_data_list)
            #print(background2_data_list)
            #print(background1_data_list, background2_data_list)
            extrabackground_data_list = get_data(rebinned_combined_background)
            real_data_list = get_data(rebinned_data)


            # #migration matrix
            # migration_name = histogram_name[avaE] + migration_matrix
            # migration_hist = file.Get(migration_name)
            # project_migration =  migration_hist.ProjectionY("migration", y_bin_min, y_bin_max)
            # migration_data = get_data(project_migration)
            # #print(migration_data)

            # data_total_value = 0

            # for data in migration_data:
            #     data_total_value = data_total_value + data

            # if data_total_value == 0:
            #     normalized_migration = [0,0,0,0,0,0,0,0]
            # else: 
            #     normalized_migration = []
            #     for data in migration_data: 
            #         normalized_migration.append(data/data_total_value)
            # #/data_total_value
            # #print(normalized_migration)
            
            # #Delete the last two terms of the list
            # normalized_migration = normalized_migration[:-2]
            # global migration_matrix_1
            # global Coh_matrix
            # for migration in range(0,6):
            #     migration_matrix_1[hist_slice-1, migration, avaE] = normalized_migration[migration]
            for Coh in range(0,20):
                Coh_matrix[hist_slice-1, Coh, avaE] = signal_data_list[Coh]
            
            #reconstructed_Coh = []
            #for Coh in signal_data_list: 
            #    reconstruct_alpha = sum(a * b for a, b in zip(phi_values, normalized_migration)) * Coh
            #    reconstructed_Coh.append(reconstruct_alpha)
            #global alpha_list
            #alpha_list[hist_slice - 1][avaE] = sum(a * b for a, b in zip(phi_values, normalized_migration))
                       
            #normalized_migration = ", ".join(f"{num:.4f}" for num in normalized_migration)
            #print(f"This migration belongs i = {hist_slice}, j= {avaE}:", normalized_migration)
            #Store all the data into the nested disctionary
            #results[hist_slice][avaE]["Coh"] = reconstructed_Coh
            results[hist_slice][avaE]["bkg1"] = background1_data_list
            results[hist_slice][avaE]["bkg2"] = background2_data_list
            results[hist_slice][avaE]["combined_bkg"] = extrabackground_data_list
            results[hist_slice][avaE]["data"] = real_data_list
        #here it should be outside the i loop but still in the j loop 
    


    #for j in range(migration_matrix_1.shape[2]):  # loop over slices
    #    print(f"\nSlice j = {j}:")
    #    print(migration_matrix_1[:, :, j])  # fixed j, full m x k matrix
    #for j in range(Coh_matrix.shape[2]):  # loop over slices
    #    print(f"\nSlice j = {j}:")
    #    print(Coh_matrix[:, :, j])  # fixed j, full m x k matrix

    true_Coh = np.zeros((6, 20, 4))
    # for hist_slice in range(1,7):
    #     for avaE in range(0,4):
    #         print(f"for bkg1 for i ={hist_slice}, j ={avaE} is:", results[hist_slice][avaE]["bkg1"])
    #         print(f"for bkg2 for i ={hist_slice}, j ={avaE} is:", results[hist_slice][avaE]["bkg2"])
    #         print(f"for combined_bkg for i ={hist_slice}, j ={avaE} is:", results[hist_slice][avaE]["combined_bkg"])
    #         print(f"for data for i ={hist_slice}, j ={avaE} is:", results[hist_slice][avaE]["data"])
    
    inverse_migration = np.zeros((6, 6, 4))
    for slice in range(0,4):
        slice_piece = migration_matrix_2[:, :, slice]
        #print(f"j={slice}:", scipy.linalg.svdvals(slice_piece, overwrite_a=False, check_finite=True))
        inverse = np.linalg.pinv(slice_piece,rcond=1e-10)
        #print(f"migration matrix for for {slice}:", slice_piece)
        np.set_printoptions(precision=8, suppress=True)
        #print(f"inverse for {slice}:", inverse)
        #print(inverse @ slice_piece)
        inverse_migration[:,:, slice] = inverse
    for avaE_1 in range(0,4):
        inverse_slice = inverse_migration[:,:,avaE_1]
        Coh_slice = Coh_matrix[:,:,avaE_1]
        for k_value in range(0,20):
            for i_value in range(0,6):
                global true_Tpion
                true_Tpion[i_value, k_value, avaE_1] = np.sum(inverse_slice[i_value, i] * Coh_slice[i, k_value] for i in range(0,6))

    for slice in range(0,4):
       matrix = migration_matrix_2[:,:, slice]
       print(f"The folllowing is j = {slice}","matrix is:", matrix)
       inverse_matrix = inverse_migration[:,:,slice]
       print("inverse is:", inverse_matrix)
       result = matrix @ inverse_matrix
       print("matrix multiplication is:", result)


    unscaled_phi_list = [1,1,1,1,1,1]
    for j_value in range(0,4):
        for i_value_1 in range(0,6):
            for k_value_1 in range(0, 20):
                true_Coh[i_value_1, k_value_1, j_value] = sum(phi_values[m_value] * migration_matrix_2[i_value_1, m_value, j_value] * true_Tpion[m_value, k_value_1, j_value] for m_value in range(0,6))
    #for j in range(true_Coh.shape[2]):  # loop over slices
    #    print(f"\nSlice j = {j}:")
    #    print(true_Coh[:, :, j])  # fixed j, full m x k matrix
    
    for avaE_3 in range(0,4):
        slice_true_coh = true_Coh[:,:,avaE_3]
        slice_Tmjk = true_Tpion[:,:,avaE_3]
        slice_coh = Coh_matrix[:,:,avaE_3]

        np.set_printoptions(precision=8, suppress=True)
        print(f"This is for j = {avaE_3}:")
        print(f"Cijk for j ={avaE_3}:", slice_coh)
        print(f"Tmjk for j={avaE_3}:", slice_Tmjk)
        print(f"True coh for j={avaE_3}:", slice_true_coh)

    for hist_slice_1 in range(1,7):
        for avaE_2 in range(0,4):
            results[hist_slice_1][avaE_2]["Coh"] = true_Coh[hist_slice_1-1, :, avaE_2]

    #Until this point, the nested for loop is done, and we have store all the information in the nested dictionary.
    #At this point, we have to get out from the for loop and create another for loop to calculate the prediction and the 
    #corresponding chi-sqaure value.


    sheet1 = build_sheet_blocks_6x20(
        {"Coh_matrix": Coh_matrix, "true_Tpion": true_Tpion, "true_Coh": true_Coh},
        i_labels, k_labels
    )
    sheet2 = build_sheet_blocks_6x6(
        {"migration_matrix": migration_matrix_2, "inverse_migration": inverse_migration},
        i_labels, m_labels
    )


    # sheet1 = sheet1.round(6)
    # sheet2 = sheet2.round(6)

    with pd.ExcelWriter("debug_matrices.xlsx", engine="xlsxwriter") as xl:
        sheet1.to_excel(xl, index=False, sheet_name="Cijk_Tmjk_trueCoh")
        sheet2.to_excel(xl, index=False, sheet_name="Migration")


    






            
    prediction_value_matrix, data_value_matrix = get_prediction(results, inputed_parameters, tune_TH1D[1], tune_TH1D[2])
    chi_square_value = chi_square(prediction_value_matrix, data_value_matrix)

            
    #print(chi_square_value)
    total_chi_square = total_chi_square + chi_square_value
    #print(total_chi_square)
    
    return total_chi_square

version = input("Enter version number: ")
#param_names = list(inspect.signature(calculate_chi_square).parameters.keys())
#param_names = [f"x{i}" for i in range(len(params))]
#start_dict = dict(zip(param_names, params))
#phi_names = param_names[-6:]
#limit_dict = {name: (0.0, None) for name in phi_names}
#result = Minuit(calculate_chi_square, **start_dict, limits=limit_dict)

#try to resolve the parameter boundary problem
#param_names = [f"x{i}" for i in range(len(params))]
#start_vals  = dict(zip(param_names, params))


initial = params[0] if isinstance(params[0], (list, np.ndarray)) else params
extract_migration()
result = Minuit(chi2_barrier, *initial)  
result.migrad(ncall=10000)
#ncall=50
optimized_params = result.values
optimized_params_list = [optimized_params[f"x{i}"] for i in range (len(optimized_params))]
#print(alpha_list)
print("Minimized Chi-square Value:", result.fval)
print("Optimized parameters are: ", optimized_params)
print("In form of a list:", optimized_params_list)
covariance_matrix = result.errors
print("Parameters' corresponding uncertainties are:", covariance_matrix)
cova = result.covariance
#print(repr(cova))
corr = result.covariance.correlation()
np.set_printoptions(precision=4, suppress=True)
print(repr(corr))

#print(optimized_params, covariance_matrix)
#revised_alpha, revised_beta, revised_gamma, revised_exp_cons = params[0], params[1], params[2], params[3]
#alpha_uncertainty = result.errors["x0"]
#beta_uncertainty = result.errors["x1"]
#gamma_uncertainty = result.errors["x2"]
#exp_uncertainty = result.errors["x3"]
#print("The corresponding parameters are:", revised_alpha, revised_beta, revised_gamma, revised_exp_cons)
#print("The correspinding uncertainties are:", alpha_uncertainty, beta_uncertainty, gamma_uncertainty, exp_uncertainty)
#print("The corresponding parameters are:", round(revised_alpha, 2), round(revised_beta, 2),  round(revised_gamma,2), round(revised_exp_cons, 2))
#print("The correspinding uncertainties are:", round(alpha_uncertainty, 2), round(beta_uncertainty, 2),  round(gamma_uncertainty, 2), round(exp_uncertainty,2))

















for hist_slice in range(1,7):
    background1_list_unscale = []
    background2_list_unscale = []
    background1_list_scale = []
    background2_list_scale = []
    #rebin --rescale -- calculate the error -- normalization by binwidth (rescale prediction, data and error) -- plot
    for avaE in range(0,4):

        y_bin_min = hist_slice
        y_bin_max = hist_slice
        #First lets get the optimized parameters stored in the way we want.
        rows =6
        columns = 3
        optimized_params_matrix = [optimized_params[i* columns: (i+1) * columns] for i in range(rows)]
        phi_list = optimized_params[-6:]
        unscaled_phi_list = [1,1,1,1,1,1]
        #migration matrix
        migration_name = histogram_name[avaE] + migration_matrix
        migration_hist = file.Get(migration_name)
        project_migration =  migration_hist.ProjectionY("migration", y_bin_min, y_bin_max)
        migration_data = get_data(project_migration)
        #print(migration_data)

        data_total_value = 0

        for data in migration_data:
            data_total_value = data_total_value + data

        if data_total_value == 0:
            normalized_migration = [0,0,0,0,0,0]
        else: 
            normalized_migration = []
            for data in migration_data: 
                normalized_migration.append(data/data_total_value)
        #/data_total_value
        
        #Delete the last two terms of the list
        #normalized_migration = normalized_migration[:-2]
        #print(normalized_migration)
        # So the following combines the first character with the main body of the histogram names
        data_hist_name1 = histogram_name[avaE] + data_hist_name
        tune_hist_names = []
        for names in tune_hist: 
            hist_name = histogram_name[avaE] + names
            tune_hist_names.append(hist_name)

        bkg_hist_names = []
        for names1 in bkg_hist_list:
            hist_name_1 = histogram_name[avaE] + names1
            bkg_hist_names.append(hist_name_1)


        #print(combined_signal_name,combined_background1_name,
        # ,combined_data_name,combined_extra_bkg1,combined_extra_bkg2,combined_extra_bkg3,combined_extra_bkg4)
        #And here we extract the histograms from the file

        data_hist = file1.Get(data_hist_name1)
        tune_TH2D = []
        for hists in tune_hist_names:
            tune_TH2D_hist = file.Get(hists)
            tune_TH2D.append(tune_TH2D_hist)

        bkg_TH2D = []
        for hists1 in bkg_hist_names:
            bkg_TH2D_hist = file.Get(hists1)
            bkg_TH2D.append(bkg_TH2D_hist)


        # Projecting tune_TH2D histograms
        tune_TH1D = []
        for idx, TH2D_hist in enumerate(tune_TH2D):
            projection_name = f"tuned_hist{idx}"  # You can customize the naming as needed
            TH1D_hist = TH2D_hist.ProjectionX(projection_name, y_bin_min, y_bin_max)
            tune_TH1D.append(TH1D_hist)

        # Projecting bkg_TH2D histograms
        bkg_TH1D = []
        for idx, bkg_TH2D_hist in enumerate(bkg_TH2D):
            projection_name = f"extra_proj_bkg{idx+1}"  # Ensures names like extra_proj_bkg1, etc.
            extra_proj_bkg = bkg_TH2D_hist.ProjectionX(projection_name, y_bin_min, y_bin_max)
            bkg_TH1D.append(extra_proj_bkg)

        # Projecting data_hist
        data_proj = data_hist.ProjectionX("data_proj", y_bin_min, y_bin_max)

        # Combining backgrounds
        combined_background = bkg_TH1D[0].Clone("combined_background")
        for bkg_hist in bkg_TH1D[1:]:
            combined_background.Add(bkg_hist)

        #print("number of bins for data is:" + str(data_proj.GetNbinsX()))
        #Next we rebin the data and hence rescale the data.

        for idx, rebins in enumerate(tune_TH1D):
            rebinning_name = f"rebinning_hist{idx}"
            rebinned_histogram = rebin_hist(rebins, rebinning_name)
            tune_TH1D[idx] = rebinned_histogram
        rebinned_combined_background = rebin_hist(combined_background, "rebinned_combined_background")
        rebinned_data = rebin_hist(data_proj, "rebinned_data")

        #Rscaling the histograms by 4
        #for idx1, rescale in enumerate(tune_TH1D):
        #    rescaled_histogram = rescaling_histogram(rescale)
        #    tune_TH1D[idx1] = rescaled_histogram
        #Rebinned_combined_background = rescaling_histogram(rebinned_combined_background)
        #Revise the bin content with the exponential.
        #hist_exp(tune_TH1D[1], optimized_params_matrix[hist_slice-1][2])
        #hist_exp(tune_TH1D[2], optimized_params_matrix[hist_slice-1][2])


        #Apply the exponential constant to the background 1&2

        #Here we create the error of the data using the TGraphError method.
        num_points = rebinned_data.GetNbinsX()
        error = ROOT.TGraphErrors(num_points)
        data_x = []
        data_y = []
        data_y_error = []
        multi_factor = []

        num_bins = rebinned_data.GetNbinsX()
        #The ROOT bin index starts from 1 instead of 0, so while writing the for loop be careful with the index.
        for i in range(1, num_points + 1):
            x_value = rebinned_data.GetBinCenter(i)
            y_value = rebinned_data.GetBinContent(i)
            data_x.append(x_value)
            data_y.append(y_value)
        #print(data_y)
        #By assigning the values of the data, next we want to correctly calculate the value of the error of each bin.
        #First we calculate the raw error of data, which is the square root of the data point.
        for i  in range(1, num_points+1):
            bin_width = rebinned_data.GetBinWidth(i) 
            multi_factor.append(1 / bin_width)
        #Can print the multiplication factor to check the results, along with the data errors.
        #print(multi_factor)

        for i in range(len(data_x)):
            raw_error = np.sqrt(data_y[i]) * multi_factor[i]
            data_y_error.append(raw_error)
        #print("the y errors:")
        #print(data_y_error)

        #Now we have the error and the value of the data we assign them to the TGarphError object.
        for i in range(num_points):
            error.SetPoint(i, data_x[i], data_y[i])
            error.SetPointError(i, 0, data_y_error[i])




        rebinned_signal_list = []
        for m in range(6):
            signal = []
            for k_value_2 in range(0,20):
                signal.append(phi_list[m] * migration_matrix_2[hist_slice-1, m, avaE] * true_Tpion[m, k_value_2, avaE])
            rebinned_signal_list.append(signal)
        
        #rebinned_signal_list[1] = [a + b for a, b in zip(rebinned_signal_list[0], rebinned_signal_list[1])]
        #rebinned_signal_list[0] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        





        #signal_data_value = show_y_value(rebinned_signal)
        sub_signal_data_1 = show_y_value(create_TH1D(rebinned_signal_list[0],"rebinned_signal_1"))
        sub_signal_data_2 = show_y_value(create_TH1D(rebinned_signal_list[1],"rebinned_signal_2"))
        sub_signal_data_3 = show_y_value(create_TH1D(rebinned_signal_list[2],"rebinned_signal_3"))
        sub_signal_data_4 = show_y_value(create_TH1D(rebinned_signal_list[3],"rebinned_signal_4"))
        sub_signal_data_5 = show_y_value(create_TH1D(rebinned_signal_list[4],"rebinned_signal_5"))
        sub_signal_data_6 = show_y_value(create_TH1D(rebinned_signal_list[5],"rebinned_signal_6"))
        sub_signal_total = []
        for i in range(0,20):
            sub_signal_total.append(sub_signal_data_1[i] + sub_signal_data_2[i] + \
                                    sub_signal_data_3[i]+sub_signal_data_4[i]+sub_signal_data_5[i] + sub_signal_data_6[i])
        #print(f"The signal for {hist_senario[hist_slice-1]}{Eava[avaE]}", signal_data_value)
        print(f"The sum for subsignal for {hist_senario[hist_slice-1]}{Eava[avaE]}", sub_signal_total)
        print(f"Phi value: {phi_list[0]}",f"The signal_1 for {hist_senario[hist_slice-1]}{Eava[avaE]}", sub_signal_data_1)
        print(f"Phi value: {phi_list[1]}",f"The signal_2 for {hist_senario[hist_slice-1]}{Eava[avaE]}", sub_signal_data_2)
        print(f"Phi value: {phi_list[2]}",f"The signal_3 for {hist_senario[hist_slice-1]}{Eava[avaE]}", sub_signal_data_3)
        print(f"Phi value: {phi_list[3]}",f"The signal_4 for {hist_senario[hist_slice-1]}{Eava[avaE]}", sub_signal_data_4)
        print(f"Phi value: {phi_list[4]}",f"The signal_5 for {hist_senario[hist_slice-1]}{Eava[avaE]}", sub_signal_data_5)
        print(f"Phi value: {phi_list[5]}",f"The signal_6 for {hist_senario[hist_slice-1]}{Eava[avaE]}", sub_signal_data_6)




        rebinned_signal_1 = rebin_and_normalize(create_TH1D(rebinned_signal_list[0],"rebinned_signal_1"), "rebinned_signal_1")
        rebinned_signal_2 = rebin_and_normalize(create_TH1D(rebinned_signal_list[1],"rebinned_signal_2"), "rebinned_signal_2")
        rebinned_signal_3 = rebin_and_normalize(create_TH1D(rebinned_signal_list[2],"rebinned_signal_3"), "rebinned_signal_3")
        rebinned_signal_4 = rebin_and_normalize(create_TH1D(rebinned_signal_list[3],"rebinned_signal_4"), "rebinned_signal_4")
        rebinned_signal_5 = rebin_and_normalize(create_TH1D(rebinned_signal_list[4],"rebinned_signal_5"), "rebinned_signal_5")
        rebinned_signal_6 = rebin_and_normalize(create_TH1D(rebinned_signal_list[5],"rebinned_signal_6"), "rebinned_signal_6")
        
        rebinned_signal = normalization(tune_TH1D[0], "tune_TH1D[0]")
        
        
        #rebinned_signal_1 = normalization(tune_TH1D[0])
        #print(show_y_value(rebinned_signal_1))
        #print("1D", show_y_value(tune_TH1D[0]))
        #rebinned_signal_2 = normalization(tune_TH1D[0])
        #print("1D", show_y_value(tune_TH1D[0]))
        #rebinned_signal_3 = normalization(tune_TH1D[0])
        #print("1D", show_y_value(tune_TH1D[0]))
        #rebinned_signal_4 = normalization(tune_TH1D[0])
        #rebinned_signal_5 = normalization(tune_TH1D[0])
        #rebinned_signal_6 = normalization(tune_TH1D[0])
        # remove the phi values in the unscaled histogram plots, only the migration matrix terms
        #alpha_1 = phi_list[0] * normalized_migration[0]
        #alpha_2 = phi_list[1] * normalized_migration[1]
        #alpha_3 = phi_list[2] * normalized_migration[2]
        #alpha_4 = phi_list[3] * normalized_migration[3]
        #alpha_5 = phi_list[4] * normalized_migration[4]
        #alpha_6 = phi_list[5] * normalized_migration[5]
        #print("normalzied migration is: ", normalized_migration)
        #print("phi list is:", phi_list)
        #print("alphas are: ", alpha_1, alpha_2, alpha_3, alpha_4, alpha_5, alpha_6)
        
        
        #print(show_y_value(rebinned_signal_1))
        #rebinened_signal_1 = rescaling_histogram_1(rebinned_signal_1, alpha_1)
        #print(show_y_value(rebinned_signal_1))
        #rebinened_signal_2 = rescaling_histogram_1(rebinned_signal_2, alpha_2)
        #rebinened_signal_3 = rescaling_histogram_1(rebinned_signal_3, alpha_3)
        #rebinened_signal_4 = rescaling_histogram_1(rebinned_signal_4, alpha_4)
        #rebinened_signal_5 = rescaling_histogram_1(rebinned_signal_5, alpha_5)
        #rebinened_signal_6 = rescaling_histogram_1(rebinned_signal_6, alpha_6)

        
        #rebinned_signal_1 = [x * phi_list[0] * normalized_migration[0] for x in rebinned_signal]
        #rebinned_signal_2 = [x * phi_list[1] * normalized_migration[1] for x in rebinned_signal]
        #rebinned_signal_3 = [x * phi_list[2] * normalized_migration[2] for x in rebinned_signal]
        #rebinned_signal_4 = [x * phi_list[3] * normalized_migration[3] for x in rebinned_signal]
        #rebinned_signal_5 = [x * phi_list[4] * normalized_migration[4] for x in rebinned_signal]
        #rebinned_signal_6 = [x * phi_list[5] * normalized_migration[5] for x in rebinned_signal]


        #print("after the normalization")
        #show_y_value(rebinned_signal)

        #print(f"the plotting extraction for bkg1 for  i ={hist_slice}, j ={avaE} is:", show_y_value(tune_TH1D[1]))
        #print(f"the plotting extraction for bkg2 for  i ={hist_slice}, j ={avaE} is:", show_y_value(tune_TH1D[2]))
        #print(f"the plotting extraction for combined_bkg for  i ={hist_slice}, j ={avaE} is:", show_y_value(rebinned_combined_background))

        rebinned_background1 = normalization(tune_TH1D[1],"tune_TH1D[1]")
        background1_list_unscale.append(rebinned_background1)
        rebinned_background2 = normalization(tune_TH1D[2],"tune_TH1D[2]")
        background2_list_unscale.append(rebinned_background2)
        rebinned_combined_background = normalization(rebinned_combined_background,"combined_background")
        rebinned_data = normalization(rebinned_data,"rebinned_data")

        for i in range(len(data_y)):
             data_y[i] = rebinned_data.GetBinContent(i+1)
             error.SetPoint(i, data_x[i], data_y[i])
        #print("the y values")
        #print(data_y)



        canvas_name = f"canvas_unscaled_{i+1}"
        canvas_title = f"Unscaled Stacked Histograms - {hist_senario[hist_slice - 1]}"
        canvas1 = ROOT.TCanvas(canvas_name, canvas_title, 1200, 800)
        stack_title = (
            f"Unscaled Histograms - {hist_senario[hist_slice - 1]} - "
            f"{Eava[avaE]}"
        )
        stack = ROOT.THStack("stack", stack_title)


        #Signal value debug





        rebinned_background1.SetLineColor(ROOT.kGreen+2)
        rebinned_background1.SetFillColor(ROOT.kGreen+2)
        rebinned_background1.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_background1)

        rebinned_background2.SetLineColor(ROOT.kGreen+3)
        rebinned_background2.SetFillColor(ROOT.kGreen+3)
        rebinned_background2.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_background2)
        
        rebinned_combined_background.SetLineColor(ROOT.kCyan+1)
        rebinned_combined_background.SetFillColor(ROOT.kCyan+1)
        rebinned_combined_background.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_combined_background)


        rebinned_signal_1.SetLineColor(ROOT.kOrange)
        rebinned_signal_1.SetFillColor(ROOT.kOrange)
        rebinned_signal_1.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_signal_1)
        rebinned_signal_2.SetLineColor(ROOT.kOrange+1)
        rebinned_signal_2.SetFillColor(ROOT.kOrange+1)
        rebinned_signal_2.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_signal_2)
        rebinned_signal_3.SetLineColor(ROOT.kOrange+2)
        rebinned_signal_3.SetFillColor(ROOT.kOrange+2)
        rebinned_signal_3.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_signal_3)
        rebinned_signal_4.SetLineColor(ROOT.kOrange+3)
        rebinned_signal_4.SetFillColor(ROOT.kOrange+3)
        rebinned_signal_4.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_signal_4)
        rebinned_signal_5.SetLineColor(ROOT.kOrange+4)
        rebinned_signal_5.SetFillColor(ROOT.kOrange+4)
        rebinned_signal_5.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_signal_5)
        rebinned_signal_6.SetLineColor(ROOT.kOrange+5)
        rebinned_signal_6.SetFillColor(ROOT.kOrange+5)
        rebinned_signal_6.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_signal_6)
        stack.Draw("HIST")

        stack.GetXaxis().SetTitle("|t| / (GeV/c)^{2}")
        stack.GetYaxis().SetTitle("Events / (GeV/c)^{2}")

        max_background = max(rebinned_background1.GetMaximum(), rebinned_background2.GetMaximum(), \
                            rebinned_combined_background.GetMaximum(),    )
        max_signal = max(rebinned_signal_1.GetMaximum(), rebinned_signal_2.GetMaximum(), rebinned_signal_3.GetMaximum(), rebinned_signal_4.GetMaximum(),\
                          rebinned_signal_5.GetMaximum(), rebinned_signal_6.GetMaximum() )
        max_data = rebinned_data.GetMaximum()

        stack.SetMaximum(max(max_background, max_signal, max_data) * 1.6)  
        stack.SetMinimum(0)

        error.SetMarkerStyle(2)  # '2' corresponds to the "plus" marker style in ROOT
        error.SetMarkerColor(ROOT.kBlack)     # Black for data points
        error.SetLineColor(ROOT.kBlack)       # Line color as black
        error.SetMarkerSize(1.5)              # Adjust marker size to make crosses more visible
        error.Draw("P SAME")

        legend = ROOT.TLegend(0.5, 0.5, 0.7, 0.7)
        legend.AddEntry(rebinned_signal_1, "Coherent (Phi_1)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_2, "Coherent (Phi_2)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_3, "Coherent (Phi_3)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_4, "Coherent (Phi_4)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_5, "Coherent (Phi_5)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_6, "Coherent (Phi_6)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_background1, "Background1 (LProton)", "f")
        legend.AddEntry(rebinned_background2, "Background2 (LNeutron)", "f")
        legend.AddEntry(rebinned_combined_background, "Combined Background", "f")
        legend.AddEntry(rebinned_data, "Data (Crosses)", "p")  # "p" for point markers
        legend.Draw()

        

        folder_name = f"{datetime.today().strftime('%Y-%m-%d')}_Ver{version}"
        base_directory = Path('/Users/jerry/Desktop/root/Root file/results_by_date') 
        new_folder = base_directory / folder_name
        new_folder.mkdir(parents=True, exist_ok=True)
        output_path = new_folder / f"{hist_senario[hist_slice-1]}{Eava[avaE]} aUNSCALED.png"
        # Update and display the canvas
        canvas1.Modified()
        canvas1.Update()
        canvas1.SaveAs(str(output_path))
        #input(f"Scenario {i+1}: Press Enter to proceed to the next iteration...")
        # Delete the canvas to free resources
        
        canvas1.Close()
        canvas2 = ROOT.TCanvas("canvas2", "Signal and Backgrounds", 800, 600)
        rebinned_background1_clone = rebinned_background1.Clone("rebinned_background1_clone")
        rebinned_background2_clone = rebinned_background2.Clone("rebinned_background2_clone")
        rebinned_signal_clone = rebinned_signal.Clone("rebinned_signal_clone")
        overall_max = max(
        rebinned_background1_clone.GetMaximum(),
        rebinned_background2_clone.GetMaximum(),
        rebinned_signal_clone.GetMaximum()
        )
        rebinned_background1_clone.SetLineColor(ROOT.kGreen+2)
        rebinned_background1_clone.SetLineWidth(2)   # Adjust line width as needed
        rebinned_background1_clone.SetFillStyle(0)   # Disable filling

        rebinned_background2_clone.SetLineColor(ROOT.kBlue+2)
        rebinned_background2_clone.SetLineWidth(2)
        rebinned_background2_clone.SetFillStyle(0)


        rebinned_signal_clone.SetLineColor(ROOT.kOrange)
        rebinned_signal_clone.SetLineWidth(2)
        rebinned_signal_clone.SetFillStyle(0)

        overall_max *= 1.6
        rebinned_background1_clone.SetMaximum(overall_max)
        rebinned_background1_clone.SetTitle(f"Unscaled Histograms - {hist_senario[hist_slice - 1]} - "
            f"{Eava[avaE]}")
        rebinned_background1_clone.Draw("HIST")
        rebinned_background2_clone.Draw("HIST SAME")
        rebinned_signal_clone.Draw("HIST SAME")
        canvas2.Update()
        legend2 = ROOT.TLegend(0.5, 0.5, 0.7, 0.7)
        legend2.AddEntry(rebinned_background1_clone, "Background1", "l")
        legend2.AddEntry(rebinned_background2_clone, "Background2", "l")
        legend2.AddEntry(rebinned_signal_clone, "Signal", "l")
        legend2.Draw()
        folder_name_1 = "Coh_and_backgrounds"
        new_folder_1 = new_folder / folder_name_1
        new_folder_1.mkdir(parents=True, exist_ok=True)
        output_path_1 = new_folder_1 / f"{hist_senario[hist_slice-1]}{Eava[avaE]} aUnscaled_Coh_and_background.png"
        canvas2.SaveAs(str(output_path_1))
        canvas2.Close()
        




        #rebinned_signal = rescaling_histogram_alpha(rebinned_signal, alpha_list[hist_slice - 1, avaE])
        #rebinned_signal_1 = rescaling_histogram_alpha(rebinned_signal_1, alpha_list[hist_slice - 1, avaE])
        #rebinned_signal_2 = rescaling_histogram_alpha(rebinned_signal_2, alpha_list[hist_slice - 1, avaE])
        #rebinned_signal_3 = rescaling_histogram_alpha(rebinned_signal_3, alpha_list[hist_slice - 1, avaE])
        #rebinned_signal_4 = rescaling_histogram_alpha(rebinned_signal_4, alpha_list[hist_slice - 1, avaE])
        #rebinned_signal_5 = rescaling_histogram_alpha(rebinned_signal_5, alpha_list[hist_slice - 1, avaE])
        #rebinned_signal_6 = rescaling_histogram_alpha(rebinned_signal_6, alpha_list[hist_slice - 1, avaE])
        rebinned_signal_list_1 = []
        for i in range(6):
            signal = []
            for k_value_2 in range(0,20):
                signal.append(phi_list[i] * migration_matrix_2[hist_slice - 1, i, avaE] * true_Tpion[i, k_value_2, avaE])
            rebinned_signal_list_1.append(signal)
        rebinned_signal_1 = normalization(create_TH1D(rebinned_signal_list_1[0],"rebinned_signal_1"),"rebinned_signal_1")
        rebinned_signal_2 = normalization(create_TH1D(rebinned_signal_list_1[1],"rebinned_signal_2"),"rebinned_signal_2")
        rebinned_signal_3 = normalization(create_TH1D(rebinned_signal_list_1[2],"rebinned_signal_3"),"rebinned_signal_3")
        rebinned_signal_4 = normalization(create_TH1D(rebinned_signal_list_1[3],"rebinned_signal_4"),"rebinned_signal_4")
        rebinned_signal_5 = normalization(create_TH1D(rebinned_signal_list_1[4],"rebinned_signal_5"),"rebinned_signal_5")
        rebinned_signal_6 = normalization(create_TH1D(rebinned_signal_list_1[5],"rebinned_signal_6"),"rebinned_signal_6")
        rebinned_background1 = rescaling_histogram_beta(rebinned_background1, optimized_params_matrix[hist_slice-1][0])
        rebinned_background2 = rescaling_histogram_gamma(rebinned_background2, optimized_params_matrix[hist_slice-1][1])
        hist_exp(rebinned_background1, optimized_params_matrix[hist_slice-1][2])
        hist_exp(rebinned_background2, optimized_params_matrix[hist_slice-1][2])
        background1_list_scale.append(rebinned_background1)
        background2_list_scale.append(rebinned_background2)



        canvas_name = f"canvas_scaled_{avaE}"
        canvas_title = f"Scaled Stacked Histograms - {hist_senario[hist_slice - 1]}"
        canvas1 = ROOT.TCanvas(canvas_name, canvas_title, 1200, 800)
        stack_title_1 = (
            f"Scaled Histograms - {hist_senario[hist_slice - 1]} - "
            f"{Eava[avaE]}"
        )
        
        stack = ROOT.THStack("stack", stack_title_1)
        rebinned_background1.SetLineColor(ROOT.kGreen+2)
        rebinned_background1.SetFillColor(ROOT.kGreen+2)
        rebinned_background1.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_background1)

        rebinned_background2.SetLineColor(ROOT.kGreen+3)
        rebinned_background2.SetFillColor(ROOT.kGreen+3)
        rebinned_background2.SetFillStyle(1001)  # Solid fill style
        stack.Add(rebinned_background2) 
 
        rebinned_combined_background.SetLineColor(ROOT.kCyan+1) 
        rebinned_combined_background.SetFillColor(ROOT.kCyan+1) 
        rebinned_combined_background.SetFillStyle(1001)  # Solid fill style 
        stack.Add(rebinned_combined_background) 
 
 
        rebinned_signal_1.SetLineColor(ROOT.kOrange) 
        rebinned_signal_1.SetFillColor(ROOT.kOrange) 
        rebinned_signal_1.SetFillStyle(1001)  # Solid fill style 
        stack.Add(rebinned_signal_1) 
        rebinned_signal_2.SetLineColor(ROOT.kOrange+1) 
        rebinned_signal_2.SetFillColor(ROOT.kOrange+1) 
        rebinned_signal_2.SetFillStyle(1001)  # Solid fill style 
        stack.Add(rebinned_signal_2) 
        rebinned_signal_3.SetLineColor(ROOT.kOrange+2) 
        rebinned_signal_3.SetFillColor(ROOT.kOrange+2) 
        rebinned_signal_3.SetFillStyle(1001)  # Solid fill style 
        stack.Add(rebinned_signal_3) 
        rebinned_signal_4.SetLineColor(ROOT.kOrange+3) 
        rebinned_signal_4.SetFillColor(ROOT.kOrange+3) 
        rebinned_signal_4.SetFillStyle(1001)  # Solid fill style 
        stack.Add(rebinned_signal_4) 
        rebinned_signal_5.SetLineColor(ROOT.kOrange+4) 
        rebinned_signal_5.SetFillColor(ROOT.kOrange+4) 
        rebinned_signal_5.SetFillStyle(1001)  # Solid fill style 
        stack.Add(rebinned_signal_5) 
        rebinned_signal_6.SetLineColor(ROOT.kOrange+5) 
        rebinned_signal_6.SetFillColor(ROOT.kOrange+5) 
        rebinned_signal_6.SetFillStyle(1001)  # Solid fill style 
        stack.Add(rebinned_signal_6) 
        stack.Draw("HIST") 
 
 
        stack.GetXaxis().SetTitle("|t| / (GeV/c)^{2}") 
        stack.GetYaxis().SetTitle("Events / (GeV/c)^{2}") 
 
        max_background = max(rebinned_background1.GetMaximum(), rebinned_background2.GetMaximum(), \
                             rebinned_combined_background.GetMaximum()) 
        max_signal = max(rebinned_signal_1.GetMaximum(), rebinned_signal_2.GetMaximum(), rebinned_signal_3.GetMaximum(), rebinned_signal_4.GetMaximum(), \
                         rebinned_signal_5.GetMaximum(), rebinned_signal_6.GetMaximum() )
        max_data = rebinned_data.GetMaximum()

        stack.SetMaximum(max(max_background, max_signal, max_data) * 1.6)  
        stack.SetMinimum(0)

        error.SetMarkerStyle(2)  # '2' corresponds to the "plus" marker style in ROOT
        error.SetMarkerColor(ROOT.kBlack)     # Black for data points
        error.SetLineColor(ROOT.kBlack)       # Line color as black
        error.SetMarkerSize(1.5)              # Adjust marker size to make crosses more visible
        error.Draw("P SAME")

        legend = ROOT.TLegend(0.5, 0.5, 0.7, 0.7)
        legend.AddEntry(rebinned_signal_1, "Coherent (Phi_1)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_2, "Coherent (Phi_2)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_3, "Coherent (Phi_3)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_4, "Coherent (Phi_4)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_5, "Coherent (Phi_5)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_signal_6, "Coherent (Phi_6)", "f")  # "f" for fill color
        legend.AddEntry(rebinned_background1, "Background1 (LProton)", "f")
        legend.AddEntry(rebinned_background2, "Background2 (LNeutron)", "f")
        legend.AddEntry(rebinned_combined_background, "Combined Background", "f")
        legend.AddEntry(rebinned_data, "Data (Crosses)", "p")  # "p" for point markers
        legend.Draw()

        # Update and display the canvas
        output_path = new_folder / f"{hist_senario[hist_slice-1]}{Eava[avaE]} SCALED.png"
        # Update and display the canvas
        canvas1.Modified()
        canvas1.Update()
        canvas1.SaveAs(str(output_path))
        #input(f"Scenario {i+1}: Press Enter to proceed to the next iteration...")

        # Delete the canvas to free resources
        canvas1.Close()
        canvas2 = ROOT.TCanvas("canvas2", "Signal and Backgrounds", 800, 600)
        rebinned_background1_clone = rebinned_background1.Clone("rebinned_background1_clone")
        rebinned_background2_clone = rebinned_background2.Clone("rebinned_background2_clone")
        rebinned_signal_clone = rebinned_signal.Clone("rebinned_signal_clone")
        overall_max = max(
        rebinned_background1_clone.GetMaximum(),
        rebinned_background2_clone.GetMaximum(),
        rebinned_signal_clone.GetMaximum()
        )
        rebinned_background1_clone.SetLineColor(ROOT.kGreen+2)
        rebinned_background1_clone.SetLineWidth(2)   # Adjust line width as needed
        rebinned_background1_clone.SetFillStyle(0)   # Disable filling

        rebinned_background2_clone.SetLineColor(ROOT.kBlue+2)
        rebinned_background2_clone.SetLineWidth(2)
        rebinned_background2_clone.SetFillStyle(0)


        rebinned_signal_clone.SetLineColor(ROOT.kOrange)
        rebinned_signal_clone.SetLineWidth(2)
        rebinned_signal_clone.SetFillStyle(0)

        overall_max *= 1.6
        rebinned_background1_clone.SetMaximum(overall_max)
        rebinned_background1_clone.SetTitle(f"Scaled Histograms - {hist_senario[hist_slice - 1]} - "
            f"{Eava[avaE]}")
        rebinned_background1_clone.Draw("HIST")
        rebinned_background2_clone.Draw("HIST SAME")
        rebinned_signal_clone.Draw("HIST SAME")
        canvas2.Update()
        legend2 = ROOT.TLegend(0.5, 0.5, 0.7, 0.7)
        legend2.AddEntry(rebinned_background1_clone, "Background1", "l")
        legend2.AddEntry(rebinned_background2_clone, "Background2", "l")
        legend2.AddEntry(rebinned_signal_clone, "Signal", "l")
        legend2.Draw()
        output_path_1 = new_folder_1 / f"{hist_senario[hist_slice-1]}{Eava[avaE]} Scaled_Coh_and_background.png"
        canvas2.SaveAs(str(output_path_1))
        canvas2.Close()
    #outside the avaE loop but in Tpion range loop
    # background1_0_25_unscale = background1_list_unscale[0]
    # background1_25_50_unscale = background1_list_unscale[1]
    # background1_50_75_unscale = background1_list_unscale[2]
    # background1_75_150_unscale = background1_list_unscale[3]
    # background1_75_150_unscale = rescaling_histogram_1(background1_75_150_unscale, 1/3)
    # background2_0_25_unscale = background2_list_unscale[0]
    # background2_25_50_unscale = background2_list_unscale[1]
    # background2_50_75_unscale = background2_list_unscale[2]
    # background2_75_150_unscale = background2_list_unscale[3]
    # background2_75_150_unscale = rescaling_histogram_1(background2_75_150_unscale, 1/3)

    # background1_0_25_scale = background1_list_scale[0]
    # background1_25_50_scale = background1_list_scale[1]
    # background1_50_75_scale = background1_list_scale[2]
    # background1_75_150_scale = background1_list_scale[3]
    # background1_75_150_scale = rescaling_histogram_1(background1_75_150_scale, 1/3)
    # background2_0_25_scale = background2_list_scale[0]
    # background2_25_50_scale = background2_list_scale[1]
    # background2_50_75_scale = background2_list_scale[2]
    # background2_75_150_scale = background2_list_scale[3]
    # background2_75_150_scale = rescaling_histogram_1(background2_75_150_scale, 1/3)

    # folder_name_2 = "Backgrounds_accross_j_bins"
    # new_folder_2 = new_folder / folder_name_2
    # new_folder_2.mkdir(parents=True, exist_ok=True)

    # canvas3 = ROOT.TCanvas("canvas3", "Unscaled background1", 800, 600)
    # overall_max = max(
    #     background1_0_25_unscale.GetMaximum(),
    #     background1_25_50_unscale.GetMaximum(),
    #     background1_50_75_unscale.GetMaximum(),
    #     background1_75_150_unscale.GetMaximum()
    #     )
    # background1_0_25_unscale.SetLineColor(ROOT.kGreen)
    # background1_0_25_unscale.SetLineWidth(2)  
    # background1_0_25_unscale.SetFillStyle(0)   

    # background1_25_50_unscale.SetLineColor(ROOT.kGreen+1)
    # background1_25_50_unscale.SetLineWidth(2)  
    # background1_25_50_unscale.SetFillStyle(0)  

    # background1_50_75_unscale.SetLineColor(ROOT.kGreen+2)
    # background1_50_75_unscale.SetLineWidth(2)  
    # background1_50_75_unscale.SetFillStyle(0)  

    # background1_75_150_unscale.SetLineColor(ROOT.kGreen+3)
    # background1_75_150_unscale.SetLineWidth(2)  
    # background1_75_150_unscale.SetFillStyle(0)  

    # overall_max *= 1.6
    # background1_0_25_unscale.SetMaximum(overall_max)
    # background1_0_25_unscale.SetTitle(f"Unscaled Histograms - {hist_senario[hist_slice - 1]} -  background 1")
    # background1_0_25_unscale.Draw("HIST")
    # background1_25_50_unscale.Draw("HIST SAME")
    # background1_50_75_unscale.Draw("HIST SAME")
    # background1_75_150_unscale.Draw("HIST SAME")
    # canvas3.Update()
    # legend3 = ROOT.TLegend(0.5, 0.5, 0.7, 0.7)
    # legend3.AddEntry(background1_0_25_unscale, f"Eava{0}", "l")
    # legend3.AddEntry(background1_25_50_unscale, f"Eava{1}", "l")
    # legend3.AddEntry(background1_50_75_unscale, f"Eava{2}", "l")
    # legend3.AddEntry(background1_75_150_unscale, f"Eava{3}", "l")
    # legend3.Draw()
    # output_path_2 = new_folder_2 / f"{hist_senario[hist_slice-1]} aUnscaled_background1.png"
    # canvas3.SaveAs(str(output_path_2))
    # canvas3.Close()







    # canvas4 = ROOT.TCanvas("canvas4", "Scaled background1", 800, 600)
    # overall_max = max(
    #     background1_0_25_scale.GetMaximum(),
    #     background1_25_50_scale.GetMaximum(),
    #     background1_50_75_scale.GetMaximum(),
    #     background1_75_150_scale.GetMaximum()
    #     )
    # background1_0_25_scale.SetLineColor(ROOT.kGreen)
    # background1_0_25_scale.SetLineWidth(2)  
    # background1_0_25_scale.SetFillStyle(0)   

    # background1_25_50_scale.SetLineColor(ROOT.kGreen+1)
    # background1_25_50_scale.SetLineWidth(2)  
    # background1_25_50_scale.SetFillStyle(0)  

    # background1_50_75_scale.SetLineColor(ROOT.kGreen+2)
    # background1_50_75_scale.SetLineWidth(2)  
    # background1_50_75_scale.SetFillStyle(0)  

    # background1_75_150_scale.SetLineColor(ROOT.kGreen+3)
    # background1_75_150_scale.SetLineWidth(2)  
    # background1_75_150_scale.SetFillStyle(0)  

    # overall_max *= 1.6
    # background1_0_25_scale.SetMaximum(overall_max)
    # background1_0_25_scale.SetTitle(f"Scaled Histograms - {hist_senario[hist_slice - 1]} -  background 1")
    # background1_0_25_scale.Draw("HIST")
    # background1_25_50_scale.Draw("HIST SAME")
    # background1_50_75_scale.Draw("HIST SAME")
    # background1_75_150_scale.Draw("HIST SAME")
    # canvas4.Update()
    # legend4 = ROOT.TLegend(0.5, 0.5, 0.7, 0.7)
    # legend4.AddEntry(background1_0_25_scale, f"Eava{0}", "l")
    # legend4.AddEntry(background1_25_50_scale, f"Eava{1}", "l")
    # legend4.AddEntry(background1_50_75_scale, f"Eava{2}", "l")
    # legend4.AddEntry(background1_75_150_scale, f"Eava{3}", "l")
    # legend4.Draw()
    # output_path_2 = new_folder_2 / f"{hist_senario[hist_slice-1]} Scaled_background1.png"
    # canvas4.SaveAs(str(output_path_2))
    # canvas4.Close()








    # canvas5 = ROOT.TCanvas("canvas5", "Unscaled background2", 800, 600)
    # overall_max = max(
    #     background2_0_25_unscale.GetMaximum(),
    #     background2_25_50_unscale.GetMaximum(),
    #     background2_50_75_unscale.GetMaximum(),
    #     background2_75_150_unscale.GetMaximum()
    #     )
    # background2_0_25_unscale.SetLineColor(ROOT.kGreen)
    # background2_0_25_unscale.SetLineWidth(2)  
    # background2_0_25_unscale.SetFillStyle(0)   

    # background2_25_50_unscale.SetLineColor(ROOT.kGreen+1)
    # background2_25_50_unscale.SetLineWidth(2)  
    # background2_25_50_unscale.SetFillStyle(0)  

    # background2_50_75_unscale.SetLineColor(ROOT.kGreen+2)
    # background2_50_75_unscale.SetLineWidth(2)  
    # background2_50_75_unscale.SetFillStyle(0)  

    # background2_75_150_unscale.SetLineColor(ROOT.kGreen+3)
    # background2_75_150_unscale.SetLineWidth(2)  
    # background2_75_150_unscale.SetFillStyle(0)  

    # overall_max *= 1.6
    # background2_0_25_unscale.SetMaximum(overall_max)
    # background2_0_25_unscale.SetTitle(f"Unscaled Histograms - {hist_senario[hist_slice - 1]} -  background 2")
    # background2_0_25_unscale.Draw("HIST")
    # background2_25_50_unscale.Draw("HIST SAME")
    # background2_50_75_unscale.Draw("HIST SAME")
    # background2_75_150_unscale.Draw("HIST SAME")
    # canvas5.Update()
    # legend5 = ROOT.TLegend(0.5, 0.5, 0.7, 0.7)
    # legend5.AddEntry(background2_0_25_unscale, f"Eava{0}", "l")
    # legend5.AddEntry(background2_25_50_unscale, f"Eava{1}", "l")
    # legend5.AddEntry(background2_50_75_unscale, f"Eava{2}", "l")
    # legend5.AddEntry(background2_75_150_unscale, f"Eava{3}", "l")
    # legend5.Draw()
    # output_path_2 = new_folder_2 / f"{hist_senario[hist_slice-1]} aUnscaled_background2.png"
    # canvas5.SaveAs(str(output_path_2))
    # canvas5.Close()




    # canvas6 = ROOT.TCanvas("canvas5", "Scaled background2", 800, 600)
    # overall_max = max(
    #     background2_0_25_scale.GetMaximum(),
    #     background2_25_50_scale.GetMaximum(),
    #     background2_50_75_scale.GetMaximum(),
    #     background2_75_150_scale.GetMaximum()
    #     )
    # background2_0_25_scale.SetLineColor(ROOT.kGreen)
    # background2_0_25_scale.SetLineWidth(2)  
    # background2_0_25_scale.SetFillStyle(0)   

    # background2_25_50_scale.SetLineColor(ROOT.kGreen+1)
    # background2_25_50_scale.SetLineWidth(2)  
    # background2_25_50_scale.SetFillStyle(0)  

    # background2_50_75_scale.SetLineColor(ROOT.kGreen+2)
    # background2_50_75_scale.SetLineWidth(2)  
    # background2_50_75_scale.SetFillStyle(0)  

    # background2_75_150_scale.SetLineColor(ROOT.kGreen+3)
    # background2_75_150_scale.SetLineWidth(2)  
    # background2_75_150_scale.SetFillStyle(0)  

    # overall_max *= 1.6
    # background2_0_25_scale.SetMaximum(overall_max)
    # background2_0_25_scale.SetTitle(f"Scaled Histograms - {hist_senario[hist_slice - 1]} -  background 2")
    # background2_0_25_scale.Draw("HIST")
    # background2_25_50_scale.Draw("HIST SAME")
    # background2_50_75_scale.Draw("HIST SAME")
    # background2_75_150_scale.Draw("HIST SAME")
    # canvas6.Update()
    # legend6 = ROOT.TLegend(0.5, 0.5, 0.7, 0.7)
    # legend6.AddEntry(background2_0_25_scale, f"Eava{0}", "l")
    # legend6.AddEntry(background2_25_50_scale, f"Eava{1}", "l")
    # legend6.AddEntry(background2_50_75_scale, f"Eava{2}", "l")
    # legend6.AddEntry(background2_75_150_scale, f"Eava{3}", "l")
    # legend6.Draw()
    # output_path_2 = new_folder_2 / f"{hist_senario[hist_slice-1]} Scaled_background2.png"
    # canvas6.SaveAs(str(output_path_2))
    # canvas6.Close()
