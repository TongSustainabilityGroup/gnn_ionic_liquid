import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops
import biosteam as bst
from biosteam import report
bst.nbtutorial()


class PET_Glycolysis_Reactor(bst.CSTR):
    _N_ins = 2
    _N_outs = 1

    def _setup(self):
        super()._setup()
        chemicals = self.chemicals
        self.amb_heat_loss = bst.Reaction('EthyleneGlycol + PET -> BHET', 'PET', self.conv, chemicals)

    def _run(self):
        effluent = self.outs[0]
        effluent.mix_from(self.ins, energy_balance=True)
        self.amb_heat_loss(effluent)
        effluent.T = self.T
        effluent.P = self.P

chemicals = bst.Chemicals(
        [
        'EthyleneGlycol', # C2H6O2
        bst.Chemical(
            'PET',
            CAS='25038-59-9',
            Cp=1.03, # Heat capacity [kJ/kg]
            rho=1350, # Density [kg/m3]
            mu=1,
            default=True,
            search_db=False,
            phase='s',
            formula="C10H8O4",
            Hf=-10000), # dummy
        bst.Chemical(
            'BHET',
            CAS='959-26-2',
            Cp=1.03, # Heat capacity [kJ/kg]
            rho=1300, # Density [kg/m3]
            mu=1,
            default=True, 
            search_db=True, 
            phase='l',
            formula="C12H14O6",
            Hf=-10000), # dummy
        '959-26-2', # BHET C12H14O6
    ])

bst.settings.set_thermo(chemicals, cache=None)
bst.settings.CEPCI = 801 # chemical engineering plant cost index 2023
steam = bst.settings.get_heating_agent('medium_pressure_steam')
steam.T = 212.37 + 273.15 # 485.52K
steam.P = 2e6

solvent_recycle = 0.9 # fraction
PET_spec = 100 # kg/hr
CO2e_elec = 0.389 # kg/kWhr 
CO2e_steam = 6.293e-05 # kg/kJ
plastic = bst.Stream('plastic', PET=PET_spec, units='kg/hr', phase='s')
solvent =  bst.Stream('solvent', EthyleneGlycol=1000, units='kg/hr')
HX1 = bst.units.HXprocess('HX1', ins=('HX1_Cin', 'HX1_Hin'), outs=('HX1_Cout', 'HX1_Hout'))
P1 = bst.units.Pump('P1', ins='P1_in', outs='P1_out', P=4e5)
E1 = bst.MultiEffectEvaporator('E1', ins='E1_in', outs=('effluent', 'sol_regen'),
                            V=solvent_recycle, V_definition='Overall', chemical='EthyleneGlycol',
                            P=(101325, 53581, 30892))
M1 = bst.units.Mixer('M1', ins=('M1_in1', 'M1_in2'), outs='M1_out', )
R1 = PET_Glycolysis_Reactor('R1', ins=('R1_in', 'solvent'), outs=('R1_out'), tau=0.15, P=101325, T=175+273.15, batch=1, tau_0=0.5)
P1.ins[:] = [solvent]
HX1.ins[:] = [P1.outs[0], E1.outs[0]]
R1.ins[:] = [plastic, M1.outs[0]]
R1.outs[:] = [E1.ins[0]]
M1.ins[:] = [HX1.outs[0], E1.outs[1]]
flowsheet_sys = bst.main_flowsheet.create_system('flowsheet_sys')

def GetUtils(tau=0.5, T=190, ot=8000, sol=10):
    Hf1 = 9000e3 / (190 - 25) * (T - 25) / (100e3 * 0.15 / 192.16812) - 460000. - 10000. 
    chemicals = bst.Chemicals(
        [
        'EthyleneGlycol', 
        bst.Chemical(
            'PET',
            CAS='25038-59-9',
            Cp=1.03, 
            rho=1350, 
            mu=1,
            default=True, 
            search_db=False, 
            phase='s',
            formula="C10H8O4",
            Hf=-10000),  # dummy
        bst.Chemical(
            'BHET',
            CAS='959-26-2',
            Cp=1.03, 
            rho=1300, 
            mu=1,
            default=True, 
            search_db=True, 
            phase='l',
            formula="C12H14O6",
            Hf=Hf1), 
        '959-26-2',
    ])
    bst.settings.set_thermo(chemicals, cache=None)    
    fresh_solvent_spec = (1 - solvent_recycle) * PET_spec * sol
    solvent.mass[0] = fresh_solvent_spec
    R1.T = T+273.15
    R1.tau = tau
    R1.conv = 0.15 * tau / 0.5 # base conversion = 0.15 for 0.5hr

    @M1.add_specification(run=True)
    def adjust_fresh_flow():
        M1.outs[0].imass['EthyleneGlycol'] = PET_spec * sol   
        M1.ins[0].imass['EthyleneGlycol'] = solvent.imass['EthyleneGlycol'] 
        M1.ins[1].imass['EthyleneGlycol'] = M1.outs[0].imass['EthyleneGlycol'] - M1.ins[0].imass['EthyleneGlycol']
    M1.simulate()
    flowsheet_sys.simulate()
    flowsheet_sys.operating_hours = ot
    
    heat_duties = [i.duty for i in flowsheet_sys.heat_utilities] 
    heat_costs = [i.cost for i in flowsheet_sys.heat_utilities]  
    steam_idx = np.argmax(heat_duties)
    water_idx = int(1 - steam_idx)
    steam = heat_duties[steam_idx] # kJ/hr
    steam_cost = heat_costs[steam_idx] * ot # USD / hr * operating hr
    water = heat_duties[water_idx] # kJ/hr
    water_cost = heat_costs[water_idx] * ot # USD / hr * operating hr
    heating_cost = steam_cost + water_cost
    
    elec = flowsheet_sys.power_utility.power # KW
    elec_cost = flowsheet_sys.power_utility.cost * ot # USD / hr * operating hr
    CO2e = ot * (CO2e_elec *  elec + CO2e_steam * steam) # kg/target tonBHET ~ kg/yr
    util_cost = flowsheet_sys.utility_cost # USD/yr ~ USD/tonBHET
    return util_cost, heating_cost, elec_cost, CO2e # USD/tonBHET, KJ/hr, KW, kg/1000 tonBHET 

def eval_costs(raw_inputs):
    smiles, cat, sol, T, t, type, size, cat_price, recycle, y = raw_inputs    
    batches = 1000 / (y * (1 / 192.2) * 252.4 * 0.1) # 1000 ton BHET / (ton BHET / batch (0.1 ton PET)) = batches 
    ot = batches * t/60 # hours operating time = batches * time/batch

    sol_price = 550 # EG USD/ton 
    BHET_price = 660 # USD/ton
    BHET_sell = BHET_price * 1000
    
    sol_cost = batches * 0.1 * sol * sol_price * 0.1 / recycle # #batch * PET/batch * sol multiplier * sol cost / reuse times 
    cat_cost = batches * 0.1 * cat * cat_price / recycle # #batch * PET ton/batch * cat multiplier * cat cost / reuse times
    util_cost, heating_cost, elec_cost, CO2e = GetUtils(tau=t/60, T=T, ot=ot, sol=sol)
    CO2e = CO2e / 1000 + (sol_cost / sol_price * 1.418 + cat_cost / cat_price * 4) 
    tot_cost =  sol_cost + cat_cost + util_cost
    return tot_cost.item(), CO2e.item(), sol_cost.item(), cat_cost.item(), util_cost.item(), heating_cost.item(), elec_cost.item(), BHET_sell, y.item()

refdf = pd.read_excel('PET_IL_Data.xlsx')
df_idx = refdf['yield'].isna()
refdf.loc[df_idx, 'yield'] = refdf[df_idx]['conversion'] * refdf[df_idx]['selectivity'] / 100
refdf = refdf.loc[refdf.loc[:,'yield'] > 0 ]

ani_price = {
    'ZnCl3': 1200,
    'CoCl4': 6800,
    'CrCl4': 10000,
    'CuCl3': 8200,
    'CuCl4': 8200,
    'Ala': 3900,
    'Ser': 3400,
    'Ac': 386,
    'Gly': 2000,
    'For': 535,
    'Asp': 6800,
    'MnCl3': 1400,
    'Co(Ac)3': 9000,
    'CoCl3': 6900,
    'Cl': 0,
    'Br':0,
    'Lys': 1270,
    'FeCl4': 2000,
    'ZnCl4': 1200,
    'Zn(Ac)3': 2100,
    'OH': 0,
    'Fe2Cl6O': 2000,
    'Pro': 18000,
    'Cu(Ac)3': 5110,
    'PO4': 870,
    'His': 44000,
    'Leu': 8700,
    'NiCl4': 6700,
    'Arg': 9400,
    'Mn(Ac)3': 2600,
    'Ni(Ac)3': 9670,
    'Try': 9500,
    'But': 1321,
    'HCO3': 250,
    'HSO4': 250,
    'Im': 8343,
    'Mesy': 1600
    }
cat_price = {
    'AMIM': 2760,
    'C6TMG': 8000, 
    'Ch': 3000,
    'C2TMG': 6000, 
    'TMG': 5000, 
    'N2222': 8290,
    'N1111': 4000,
    'C4TMG': 7000, 
    'C8TMG': 9000, 
    'BMIM': 2000,
    'HMIM': 2760,
    'DMIM': 2600,
    'DEIM': 2600,
    'EMIM': 2300,
    'UREA': 300,
}
cation_list = np.unique(refdf[['cation_name','cation_smiles']].values.astype(str),axis=0)
cation_list = np.char.replace(cation_list, ' ', '')
cation_dict = dict(zip(cation_list[:,0], cation_list[:,1]))
anion_list =  np.unique(refdf[['anion_name','anion_smiles']].values.astype(str),axis=0)
anion_list =  np.char.replace(anion_list, ' ', '')
anion_dict = dict(zip(anion_list[:,0], anion_list[:,1]))
mol_dict = {}
for i in cation_dict.keys():
    i_mol = Chem.MolFromSmiles(cation_dict[i])
    mol_dict[i] = (i, cation_dict[i], rdmolops.GetFormalCharge(i_mol), Descriptors.MolWt(i_mol), cat_price[i], 1)
for i in anion_dict.keys():
    i_mol = Chem.MolFromSmiles(anion_dict[i])
    mol_dict[i] = (i, anion_dict[i], rdmolops.GetFormalCharge(i_mol), Descriptors.MolWt(i_mol), ani_price[i], 0)

sndata_exp = pd.read_excel('PET_IL_Data_exp.xlsx')
# sndata_exp = pd.read_csv('screen_data.csv')

def calc_price_exp(n):
    i = (sndata_exp.cation_name.iloc[n], sndata_exp.anion_name.iloc[n])
    ic = mol_dict[i[0]]
    ia = mol_dict[i[1]]
    ir = - ia[2] / ic[2] 
    mc = ir * ic[3] 
    ma = ia[3]
    pc = mc / (mc + ma) * ic[-2]
    pa = ma / (mc + ma) * ia[-2]
    p = pc + pa
    return round(p)

sndata_exp['catalyst_price'] = [calc_price_exp(j) for j in range(len(sndata_exp))]
sndata_exp['recycle_times'] = 1
sndata_exp['yield'] /= 100
screen_data = sndata_exp[['IL_smiles', 'catalyst_amount', 'solvent_amount', 'temperature_c', 
                       'reaction_time_min', 'PET_source', 'PET_size_mm', 'catalyst_price',
                       'recycle_times', 'yield']]

anslist = np.zeros([len(screen_data) ,5])
for i in range(len(screen_data)):
    a = screen_data.iloc[i].to_list()
    ans = eval_costs(a)
    anslist[i] = [ans[0]/1000, ans[1]/1000, ans[2]/1000, ans[3]/1000, ans[4]/1000] 
        
    if np.mod(i, 1050) == 0:
        print(i)
        
        bst.main_flowsheet.clear()
        chemicals = bst.Chemicals(
                [
                'EthyleneGlycol', 
                bst.Chemical(
                    'PET',
                    CAS='25038-59-9',
                    Cp=1.03, 
                    rho=1350, 
                    mu=1,
                    default=True, 
                    search_db=False, 
                    phase='s',
                    formula="C10H8O4",
                    Hf=-10000), 
                bst.Chemical(
                    'BHET',
                    CAS='959-26-2',
                    Cp=1.03, 
                    rho=1300,
                    mu=1,
                    default=True, 
                    search_db=True, 
                    phase='l',
                    formula="C12H14O6",
                    Hf=-10000), 
                '959-26-2',
            ])
        bst.settings.set_thermo(chemicals, cache=None)
        bst.settings.CEPCI = 801 
        steam = bst.settings.get_heating_agent('medium_pressure_steam')
        steam.T = 212.37 + 273.15 
        steam.P = 2e6

        solvent_recycle = 0.9 
        PET_spec = 100 
        CO2e_elec = 0.389 
        CO2e_steam = 6.293e-05
        plastic = bst.Stream('plastic', PET=PET_spec, units='kg/hr', phase='s')
        solvent =  bst.Stream('solvent', EthyleneGlycol=1000, units='kg/hr')
        HX1 = bst.units.HXprocess('HX1', ins=('HX1_Cin', 'HX1_Hin'), outs=('HX1_Cout', 'HX1_Hout'))
        P1 = bst.units.Pump('P1', ins='P1_in', outs='P1_out', P=4e5)
        E1 = bst.MultiEffectEvaporator('E1', ins='E1_in', outs=('effluent', 'sol_regen'),
                                    V=solvent_recycle, V_definition='Overall', chemical='EthyleneGlycol',
                                    P=(101325, 53581, 30892))
        M1 = bst.units.Mixer('M1', ins=('M1_in1', 'M1_in2'), outs='M1_out', )
        R1 = PET_Glycolysis_Reactor('R1', ins=('R1_in', 'solvent'), outs=('R1_out'), tau=0.15, P=101325, T=175+273.15, batch=1, tau_0=0.5)
        P1.ins[:] = [solvent]
        HX1.ins[:] = [P1.outs[0], E1.outs[0]]
        R1.ins[:] = [plastic, M1.outs[0]]
        R1.outs[:] = [E1.ins[0]]
        M1.ins[:] = [HX1.outs[0], E1.outs[1]]
        flowsheet_sys = bst.main_flowsheet.create_system('flowsheet_sys')

pd.DataFrame(anslist).to_excel("output.xlsx")