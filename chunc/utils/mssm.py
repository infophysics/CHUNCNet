"""
Interface to the micrOMEGAs software
"""
from multiprocessing.sharedctypes import Value
import numpy as np
import scipy as sp
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import csv
import os
from datetime import datetime
from chunc.dataset.parameters import *

"""
Standard model inputs for SoftSUSY.
1) alpha^(-1) : inverse electromagnetic coupling
2) G_F        : Fermi constant
3) alpha_s    : strong coupling at the Z pole
4) m_Z        : pole mass
5) m_b        : b quark running mass
6) m_t        : pole mass
7) m_tau      : pole mass
"""
sm_inputs = [
    ["BLOCK SMINPUTS"],
    [" 1 1.27934e+02"],     
    [" 2 1.16637e-05"],
    [" 3 1.17200e-01"],
    [" 4 9.11876e+01"],
    [" 5 4.25000e+00"],
    [" 6 1.74200e+02"],  
    [" 7 1.77700e+00"],
]

fail_values = [-1 for ii in range(146)]

class MSSMGenerator:
    """
    This class is responsible for running CMSSM and PMSSM
    parameters through SoftSUSY and micrOMEGAs to calculate
    SM parameter values (like the higgs mass and dm relic density).
    It requires that you've installed both SoftSUSY and micrOMEGAs
    and must provide the location of their directories.
    """
    def __init__(self,
        microemgas_dir: str,
        softsusy_dir:   str,
        param_space:    str='cmssm'
    ):
        # set micromegas and softsusy directories
        self.micromegas_dir = microemgas_dir
        self.softsusy_dir   = softsusy_dir
        # set the parameter space
        if param_space not in ['simple', 'cmssm', 'pmssm']:
            raise ValueError(f"Specified parameter space '{param_space}' not allowed!")
        self.param_space    = param_space
        # set up input variable names
        if self.param_space == 'simple':
            self.input_variables = simple_columns
        elif self.param_space == 'cmssm':
            self.input_variables = cmssm_columns
        else:
            self.input_variables = pmssm_columns
        
        # various commands
        self.softsusy_cmd = self.softsusy_dir + "/softpoint.x leshouches < "
        self.micromegas_cmd = self.micromegas_dir + "/main "

        # indices of events which are "super-invalid"
        self.super_invalid = []

    def load_file(self,
        input_file: str,
    ):
        """
        This simple function loads in parameter values
        for both CMSSM and PMSSM theories.
        """
        mssm_points = pd.read_csv(
            input_file, 
            names=self.input_variables,
            usecols=self.input_variables,
            dtype=float
        )
        return mssm_points
    
    def sugra_input(self,
        m_scalar:     float=125.0,
        m_gaugino:    float=500.0,
        trilinear:    float=0.0,
        higgs_tanbeta:float=10.0,
        sign_mu:      float=1.0,
    ):
        """
        This function generates the input for SoftSUSY.
        The default inputs to this model are from the 
        CMSSM10.1.1.
        """
        sugra = [["BLOCK MODSEL"]]
        sugra.append([" 1 1"])          # mSUGRA model
        sugra.append([" 11 32"])        # number of log-spaced grid points
        sugra.append([" 12 10.000e+17"])   # largest Q scale
        sugra += sm_inputs
        sugra.append(["BLOCK MINPAR"])
        sugra.append([f" 3 {higgs_tanbeta:.6e}"])
        sugra.append([f" 4 {int(sign_mu)}"])
        sugra.append([f" 5 {trilinear:.6e}"])
        sugra.append([f"BLOCK EXTPAR"])
        # Gaugino masses
        sugra.append([f" 1 {m_gaugino:.6e}"])   # Bino mass
        sugra.append([f" 2 {m_gaugino:.6e}"])   # Wino mass
        sugra.append([f" 3 {m_gaugino:.6e}"])   # gluino mass
        # trilinear couplings
        sugra.append([f" 11 {trilinear:.6e}"])  # Top trilinear coupling
        sugra.append([f" 12 {trilinear:.6e}"])  # Bottom trilinear coupling
        sugra.append([f" 13 {trilinear:.6e}"])  # tau trilinear coupling
        # Higgs parameters
        sugra.append([f" 21 {pow(m_scalar,2):.6e}"]) # down type higgs mass^2
        sugra.append([f" 22 {pow(m_scalar,2):.6e}"]) # up type higgs mass^2
        # sfermion masses
        sugra.append([f" 31 {m_scalar:.6e}"])   # left 1st-gen scalar lepton
        sugra.append([f" 32 {m_scalar:.6e}"])   # left 2nd-gen scalar lepton
        sugra.append([f" 33 {m_scalar:.6e}"])   # left 3rd-gen scalar lepton
        sugra.append([f" 34 {m_scalar:.6e}"])   # right scalar electron mass
        sugra.append([f" 35 {m_scalar:.6e}"])   # right scalar muon mass
        sugra.append([f" 36 {m_scalar:.6e}"])   # right scalar tau mass
        sugra.append([f" 41 {m_scalar:.6e}"])   # left 1st-gen scalar quark
        sugra.append([f" 42 {m_scalar:.6e}"])   # left 2nd-gen scalar quark
        sugra.append([f" 43 {m_scalar:.6e}"])   # left 3rd-gen scalar quark
        sugra.append([f" 44 {m_scalar:.6e}"])   # right scalar up
        sugra.append([f" 45 {m_scalar:.6e}"])   # right scalar charm
        sugra.append([f" 46 {m_scalar:.6e}"])   # right scalar top
        sugra.append([f" 47 {m_scalar:.6e}"])   # right scalar down
        sugra.append([f" 48 {m_scalar:.6e}"])   # right scalar strange
        sugra.append([f" 49 {m_scalar:.6e}"])   # right scalar bottom

        return sugra

    def universal_input(self,
        m_bino:     float,
        m_wino:     float,
        m_gluino:   float,
        trilinear_top:      float,
        trilinear_bottom:   float,
        trilinear_tau:      float,
        higgs_mu:       float,
        higgs_pseudo:   float,
        m_left_electron:float,
        m_left_tau:     float,
        m_right_electron:   float,
        m_right_tau:        float,
        m_scalar_quark1:    float,
        m_scalar_quark3:    float,
        m_scalar_up:        float,
        m_scalar_top:       float,
        m_scalar_down:      float,
        m_scalar_bottom:    float,
        higgs_tanbeta:      float,
    ):
        """
        This function generates the input for SoftSUSY for
        more general theories (including PMSSM).
        """
        universal = [["BLOCK MODSEL"]]
        universal.append([" 1 0"])          # Universal model
        universal.append([" 11 32"])        # number of log-spaced grid points
        universal.append([" 12 1.000e17"])    # largest Q scale
        universal += sm_inputs
        universal.append(["BLOCK MINPAR"])
        universal.append([f" 3 {higgs_tanbeta:.6e}"])
        universal.append([f"BLOCK EXTPAR"])
        # input scale
        universal.append([f" 0 -1"])    # a priori unknown input scale
        # Gaugino masses
        universal.append([f" 1 {m_bino:.6e}"])      # Bino mass
        universal.append([f" 2 {m_wino:.6e}"])      # Wino mass
        universal.append([f" 3 {m_gluino:.6e}"])    # Gluino mass
        # Trilinear couplings
        universal.append([f" 11 {trilinear_top:.6e}"])      # Top trilinear coupling
        universal.append([f" 12 {trilinear_bottom:.6e}"])   # Bottom trilinear coupling
        universal.append([f" 13 {trilinear_tau:.6e}"])      # Tau trilinear coupling
        # Higgs parameters
        universal.append([f" 23 {higgs_mu:.6e}"])       # mu parameter
        universal.append([f" 26 {higgs_pseudo:.6e}"])   # psuedoscalar higgs pole mass
        # sfermion masses
        universal.append([f" 31 {m_left_electron:.6e}"])    # left 1st-gen scalar lepton
        universal.append([f" 32 {m_left_electron:.6e}"])    # left 2nd-gen scalar lepton
        universal.append([f" 33 {m_left_tau:.6e}"])         # left 3rd-gen scalar lepton
        universal.append([f" 34 {m_right_electron:.6e}"])   # right scalar electron mass
        universal.append([f" 35 {m_right_electron:.6e}"])   # right scalar muon mass
        universal.append([f" 36 {m_right_tau:.6e}"])        # right scalar tau mass
        universal.append([f" 41 {m_scalar_quark1:.6e}"])    # left 1st-gen scalar quark
        universal.append([f" 42 {m_scalar_quark1:.6e}"])    # left 2nd-gen scalar quark
        universal.append([f" 43 {m_scalar_quark3:.6e}"])    # left 3rd-gen scalar quark
        universal.append([f" 44 {m_scalar_up:.6e}"])        # right scalar up
        universal.append([f" 45 {m_scalar_up:.6e}"])        # right scalar charm
        universal.append([f" 46 {m_scalar_top:.6e}"])       # right scalar top
        universal.append([f" 47 {m_scalar_down:.6e}"])      # right scalar down
        universal.append([f" 48 {m_scalar_down:.6e}"])      # right scalar strange
        universal.append([f" 49 {m_scalar_bottom:.6e}"])    # right scalar bottom

        return universal

    def parse_slha(self,
        event_id:   int,
    ):  
        """
        This function parses SLHA output files from SoftSUSY
        to get various parameter values.
        """
        parameters = {}
        input_values = []
        with open(f".tmp/susy_output_{event_id}", "r") as file:
            reader = csv.reader(file,delimiter='@')
            for row in reader:
                input_values.append([item for item in row])
        # loop through input values and look for blocks
        temp_block = ''
        for row in input_values:
            split_row = row[0].split()
            """
            First check if this line defines a new 'Block',
            and whether that block has an associated 'Q' value.
            """
            if split_row[0] == 'Block':
                temp_block = split_row[1]
                if temp_block not in parameters.keys():
                    parameters[temp_block] = {}
                if split_row[2] == 'Q=':
                    if 'Q' not in parameters[temp_block]:
                        parameters[temp_block]['Q'] = [round(float(split_row[3]),6)]
                    else:
                        parameters[temp_block]['Q'].append(round(float(split_row[3]),6))
                continue
            # if a comment line, then skip
            elif split_row[0] == '#':
                continue
            # Now parse the results of this block.
            elif split_row[2] == '#':
                row_type = split_row[3].replace('(Q)','').replace('MSSM','').replace('(MX)','')
                if 'Q' in parameters[temp_block]:
                    if row_type not in parameters[temp_block]:
                        parameters[temp_block][row_type] = [round(float(split_row[1]),6)]
                    else:
                        parameters[temp_block][row_type].append(round(float(split_row[1]),6))
                else:
                    try:
                        parameters[temp_block][row_type] = [round(float(split_row[1]),6)]
                    except:
                        parameters[temp_block][row_type] = [split_row[1]]
            elif split_row[3] == '#':
                row_type = split_row[4].replace('(Q)','').replace('MSSM','').replace('(MX)','')
                if 'Q' in parameters[temp_block]:
                    if row_type not in parameters[temp_block]:
                        parameters[temp_block][row_type] = [round(float(split_row[2]),6)]
                    else:
                        parameters[temp_block][row_type].append(round(float(split_row[2]),6))
                else:
                    try:
                        parameters[temp_block][row_type] = [round(float(split_row[2]),6)]
                    except:
                        parameters[temp_block][row_type] = [split_row[2]]
        return parameters

    def compute_gut_coupling(self,
        parameters: dict
    ):
        """
        This function receives input from 'parse_slha' and 
        computes various gauge couplings via linear interpolation.
        """
        gauge_couplings = [-1,-1,-1,-1]
        if self.param_space == 'pmssm':
            mx_scale = parameters['EXTPAR']['Set'][0]
        else:
            mx_scale = parameters['EXTPAR']['MX'][0]
        gauge_q      = parameters['gauge']['Q']
        gauge_gprime = parameters['gauge']["g'"]
        gauge_g      = parameters['gauge']['g']
        gauge_g3     = parameters['gauge']['g3']
        for ii in range(len(gauge_q)-1):
            if (gauge_q[ii] < mx_scale and gauge_q[ii+1] > mx_scale):
                high_q = gauge_q[ii+1]
                low_g,  high_g = gauge_g[ii], gauge_g[ii+1]
                low_g3, high_g3 = gauge_g3[ii], gauge_g3[ii+1]
                low_gprime, high_gprime = gauge_gprime[ii], gauge_gprime[ii+1]
                ratio = np.log10(mx_scale/high_q)
                # set gauge couplings
                gauge_couplings[0] = mx_scale
                gauge_couplings[1] = np.sqrt(5.0/3.0) * (low_gprime + ratio * (high_gprime - low_gprime))
                gauge_couplings[2] = low_g + ratio * (high_g - low_g)
                gauge_couplings[3] = low_g3 + ratio * (high_g3 - low_g3)
        return gauge_couplings


    def run_softsusy(self,
        event_id:   int,
    ):
        """
        This function runs softsusy for a particular event
        input which is identified by its event id.
        """
        # create the input file
        if self.param_space == 'cmssm':
            tmp_input = self.sugra_input(
                m_scalar=self.input_params.iloc[event_id]['gut_m0'],
                m_gaugino=self.input_params.iloc[event_id]['gut_m12'],
                trilinear=self.input_params.iloc[event_id]['gut_A0'],
                higgs_tanbeta=self.input_params.iloc[event_id]['gut_tanb'],
                sign_mu=self.input_params.iloc[event_id]['sign_mu']
            )
        else:
            tmp_input = self.universal_input(
                m_bino=self.input_params.iloc[event_id]['gut_m1'],
                m_wino=self.input_params.iloc[event_id]['gut_m2'],
                m_gluino=self.input_params.iloc[event_id]['gut_m3'],
                trilinear_top=self.input_params.iloc[event_id]['gut_At'],
                trilinear_bottom=self.input_params.iloc[event_id]['gut_Ab'],
                trilinear_tau=self.input_params.iloc[event_id]['gut_Atau'],
                higgs_mu=self.input_params.iloc[event_id]['gut_mmu'],
                higgs_pseudo=self.input_params.iloc[event_id]['gut_mA'],
                m_left_electron=self.input_params.iloc[event_id]['gut_mL1'],
                m_left_tau=self.input_params.iloc[event_id]['gut_mL3'],
                m_right_electron=self.input_params.iloc[event_id]['gut_me1'],
                m_right_tau=self.input_params.iloc[event_id]['gut_mtau1'],
                m_scalar_quark1=self.input_params.iloc[event_id]['gut_mQ1'],
                m_scalar_quark3=self.input_params.iloc[event_id]['gut_mQ3'],
                m_scalar_up=self.input_params.iloc[event_id]['gut_mu1'],
                m_scalar_top=self.input_params.iloc[event_id]['gut_mu3'],
                m_scalar_down=self.input_params.iloc[event_id]['gut_md1'],
                m_scalar_bottom=self.input_params.iloc[event_id]['gut_md3'],
                higgs_tanbeta=self.input_params.iloc[event_id]['gut_tanb'],
            )
        with open(f".tmp/input_{event_id}.csv", "w") as file:
            writer = csv.writer(file)
            writer.writerows(tmp_input)
        # build the susy command and run it
        cmd = self.softsusy_cmd + f" .tmp/input_{event_id}.csv > .tmp/susy_output_{event_id}"
        return os.system(cmd)

    
    def run_micromegas(self,
        event_id:   int,
    ):
        """
        This function runs the micromegas for a particular
        output slha file from softsusy.
        """
        cmd = self.micromegas_cmd + f" .tmp/susy_output_{event_id} .tmp/ _{event_id}"
        return os.system(cmd)
                
    def parse_micromegas(self,
        event_id:   int,
    ):
        """
        This function parses the micromegas for a particular
        output slha file from softsusy.
        """
        # read in results
        with open(f".tmp/micrOmegas_output_{event_id}.txt", "r") as file:
            reader = csv.reader(file,delimiter=",")
            return next(reader)

    def run_event(self,
        event_id:   int,
    ):  
        """
        Runs the SoftSUSY + micrOMEGAs software
        for a particular event.
        """
        if self.param_space == 'cmssm':
            input_params = [
                self.input_params.iloc[event_id]['gut_m0'],
                self.input_params.iloc[event_id]['gut_m12'],
                self.input_params.iloc[event_id]['gut_A0'],
                self.input_params.iloc[event_id]['gut_tanb'],
                self.input_params.iloc[event_id]['sign_mu']
            ]
        else:
            input_params = [
                self.input_params.iloc[event_id]['gut_m1'],
                self.input_params.iloc[event_id]['gut_m2'],
                self.input_params.iloc[event_id]['gut_m3'],
                self.input_params.iloc[event_id]['gut_mmu'],
                self.input_params.iloc[event_id]['gut_mA'],
                self.input_params.iloc[event_id]['gut_At'],
                self.input_params.iloc[event_id]['gut_Ab'],
                self.input_params.iloc[event_id]['gut_Atau'],
                self.input_params.iloc[event_id]['gut_mL1'],
                self.input_params.iloc[event_id]['gut_mL3'],
                self.input_params.iloc[event_id]['gut_me1'],
                self.input_params.iloc[event_id]['gut_mtau1'],
                self.input_params.iloc[event_id]['gut_mQ1'],
                self.input_params.iloc[event_id]['gut_mQ3'],
                self.input_params.iloc[event_id]['gut_mu1'],
                self.input_params.iloc[event_id]['gut_mu3'],
                self.input_params.iloc[event_id]['gut_md1'],
                self.input_params.iloc[event_id]['gut_md3'],
                self.input_params.iloc[event_id]['gut_tanb'],
            ]
        softsusy_error = self.run_softsusy(event_id)
        if (softsusy_error != 0):
            #print(f"Error occured with SoftSUSY for event {event_id}.")
            os.remove(f".tmp/input_{event_id}.csv")
            os.remove(f".tmp/susy_output_{event_id}")
            with open(".tmp/super_invalid_ids.txt", "a") as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow([event_id])
            return
        micromegas_error = self.run_micromegas(event_id)
        if (micromegas_error != 0):
            #print(f"Error occured with micrOMEGAs for event {event_id}.")
            os.remove(f".tmp/input_{event_id}.csv")
            os.remove(f".tmp/susy_output_{event_id}")
            with open(".tmp/super_invalid_ids.txt", "a") as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow([event_id])
            return
        susy_params = self.parse_slha(event_id)
        micromegas_params = self.parse_micromegas(event_id)
        gut_params = self.compute_gut_coupling(susy_params)
        physical_params = [
            susy_params[item][key][0] for ii, (key,item) in enumerate(softsusy_physical_parameters.items())
        ]
        weak_params = [
            susy_params[item][key][0] for ii, (key, item) in enumerate(softsusy_weak_parameters.items())
        ]
        all_params = input_params + physical_params + weak_params + micromegas_params + gut_params
        with open(f".tmp/final_output_{event_id}", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows([all_params])
        

    def run_parameters(self,
        input_file:     str,
        output_dir:     str='mssm_output/',
        num_events:     int=-1,
        num_workers:    int=1,
        save_super_invalid: bool=True,
    ):
        """
        The main function which runs through SoftSUSY and
        micrOMEGAs for a give set of parameter values.
        """
        # load in input parameters
        self.input_params = self.load_file(input_file)
        # make temporary dictionary
        if not os.path.isdir('.tmp/'):
            os.mkdir('.tmp/')
        if not os.path.isdir(f'{output_dir}/'):
            os.mkdir(f'{output_dir}/')
        # assign workers and iterate over
        # mssm points
        if (
            (num_events == -1) or
            (num_events > len(self.input_params))
        ):
            events = [ii for ii in range(len(self.input_params))]
        else:
            events = [ii for ii in range(num_events)]
        with Pool(num_workers) as p:
            #p.map(self.run_event, events)
            r = list(tqdm(p.imap(self.run_event, events), total=len(events)))
        final_outputs = []
        # grab the generated outputs
        for ii in range(len(events)):
            try:
                with open(f".tmp/final_output_{ii}", "r") as file:
                    reader = csv.reader(file, delimiter=",")
                    final_outputs.append(next(reader))
            except Exception as e:
                pass
        # generate final output file
        now = datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
        with open(f"{output_dir}/{self.param_space}_generated_{now}.txt", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerows(final_outputs)
        for ii in range(len(events)):
            try:
                os.remove(f".tmp/input_{ii}.csv")
                os.remove(f".tmp/susy_output_{ii}")
                os.remove(f".tmp/micrOmegas_output_{ii}.txt")
                os.remove(f".tmp/final_output_{ii}")
            except Exception as e:
                pass
        
        # save super invalid points
        if save_super_invalid:
            with open(".tmp/super_invalid_ids.txt", "r") as file:
                reader = csv.reader(file, delimiter=",")
                for row in reader:
                    self.super_invalid.append(int(row[0]))
            super_invalid_points = [
                self.input_params.iloc[ii] for ii in self.super_invalid
            ]
            with open(f"{output_dir}/{self.param_space}_super_invalid.txt", "w") as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerows(super_invalid_points)
        try:
            os.remove(".tmp/super_invalid_ids.txt")
        except Exception as e:
            pass
        
        # clean up
        if os.path.isdir(".tmp/"):
            os.rmdir(".tmp/")