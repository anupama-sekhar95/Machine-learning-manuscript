import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data1 = pd.read_csv("../data/data_to_cluster_edited.csv")
X_original = data1.drop(["Plant order", "Plant family", "Plant genus", "Species", "plant name","Plant part_Methodology"], axis=1)
y = data1['Plant order']
y_all = data1[["Plant order", "Plant family", "Plant genus", "Species", "plant name","Plant part_Methodology"]]


def strategy_base():
    # Evaluate the model with these new features
    X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.33, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Model accuracy with aggregated features: {acc:.4f}\n")
    return acc

# --- 2. STRATEGY IMPLEMENTATIONS ---
def strategy_1_aggregated_profile(data1, data2, data2_indexed, agg_method='mean'):
    """
    Strategy 1: Creates an aggregated fingerprint profile.
    Args:
        data1 (pd.DataFrame): DataFrame with target and compound presence.
        data2 (pd.DataFrame): DataFrame with compound fingerprints.
        agg_method (str): Aggregation method: 'mean', 'sum', or 'binary'.
        
    Returns:
        pd.DataFrame: The newly engineered feature matrix.
    """
    print(f"--- Running Strategy 1: Aggregated Profile (method: {agg_method}) ---")
    
    new_feature_list = []
    l=[]

    missing_compound = []
    for index, row in X_original.iterrows():
        present_compounds = row[row == 1].index.tolist()

        count = len(present_compounds)
        if not present_compounds:
            aggregated_profile = np.zeros(data2_indexed.shape[1])
            new_feature_list.append(aggregated_profile)
            continue
                    
        try:
            try:
                present_compounds[present_compounds.index('6-methyl-5- hepten-2-one')] = '6-methyl-5-hepten-2-one'
            except:pass
            compound_fingerprints = data2_indexed.loc[present_compounds]
        except:
            if 'germacrene D' in present_compounds:
                present_compounds.remove('germacrene D')
            missing_compound = ['Z-3-Hexenyl-a-methyl butyrate', 'dodecan-1,5-olide', 'Azuleno', '2,3-dimethyl-5-heptanal', 'Methyl nerylate', '10-(2-methylbutyryloxy)-8,9-didehydrothymol isobutyrate', '8,9-didehydrothymol methyl ether', 'cis-3-hexenyl tiglate', 'l-pentadecene', 'dehydrosabina ketone', 'Z-pinene hydrate', 'trans-g-cadinene', 'C15H24O', 'Heptane-1,5-olide', 'Methyl tetracosanate', 'p-cumene-8-ol', 'a-oxo-ethanol-benzoate', 'E-g-bisabolene', '9,10-Dihydroisolongifolene', 'cis-p-mentha-1(7),8-dien-2-ol', 'E-allo ocimene', 'E-p-menthan-2-ol-1,8-oxide', 'Lilac alcohol D', 'Methyl 2-methyl-2(Z)-butenoate', 'E-a-farnesene', 'Sylvestrane', '2-methyl-6-methylene-1,7-octadiene-2-one', 'Butyl octylbenzene', '5-methyl-2-hexanal', 'Linalool oxide(pyranoid; alcohol)', '8,8-dimethyl-4-methylene-1-oxoaspiro(2,5)-oct-5-ene', 'isobutyraldehyde oxime (syn)', '5-methyl-6-en-2-one', 'E-ene-yne-Dicycloether', 'Dimethyl disulphide monosulphone', 'Dehydro sabine ketone', 'mesistylene', '4a-a,7-b,7a-a-nepetalactone', 'E-2,6-DimethyInona-6,8-dien-4-ol', 'methyl benzoate. 2.5-dimethyl-', 'Z,Z-Heptadeca-8,11-dienal', 'C15H28o', '3-Methyl isobutylbutanoate', 'Methyl docosanate', 'Methyl eicosanate', '5-phenylmethoxy-1-pentanol', 'cis-limonene epoxide', 'cis-Isopulegone', '2-hydroxy piperitone', '14-Hydroxy-Z-caryophyllene', '6,7-dihydro-8-hydroxylinalool', '4-hydroxy-2-methylcyclopentan-1,3-dione', 'cis-linalool oxide acetate (pyranoid)', 'caryophylla-2(12),6-dien-5a-ol', 'Propyl nonyl benzene', 'a-p-Dimethyl styrene', '3,5-dimethyl-pyran-4-one', 'isocyanato-methylbenzene', 'Xanthorrhizol isomer', 'E-(E)-7-Mega stigmen-3,9-dione', 'trans-p-mentha-1(7),8-dien-2-ol', 'E-Geranyl acetone', 'ar-Curcumene 15-al', 'E-8-undecanal', 'Dehydroelshotzia ketone', '3,5-dimethyl methyl benzoate', 'Decahydro-4,4,8,9,10-pentamethylnaphthalene', 'pinocarveol B', '14–hydroxy–d–cadinene', ' 7,13-abietadiene', 'Pentyl-2-methyl-2-butanoate', 'Pentyl heptylbenzene', 'labda-7,14-dien-13-ol', '3-methylamyl valerate', 'abieta-7,13-diene-3-one', 'Methyl-4-methoxycyclohex-1-ene', 'cis-tent-muurola-3,5-dien', 'E-2,6-DimethyInona-3,6,8-trien-2-ol', '2E,4E-nonadienal', 'E-b-ocimene epoxide', '4,8 b-Epoxycaryophyllene', 'Z-5-Tricosene', 'p-cymen-8-yl acetate', '4-methyl pentanone', '2-pentyl propanoate', '3-methylcyclopentan-1,2-dione', 't,4-dimethylbenzene butanal', 'Methyl ester hexadecanoic acid', 'cis-a-bergamotol', '9-19-Cycloanost_x0002_6-ene-3,7-diol', 'Z-cinnamic alcohol', 'Methyl (2R)-2-acetoxy-4-methylpentanoate', 'b-Safranol', '2,2-dimethoxy propanone', 'Methyl (2S,3S)-2-acetoxy-3-methylpentanoate', '8-Acetoxypatchoulialcohol', 'Hexadecenoic acid', '3-Methyl butylhexanoate', 'Dimethyl napthalene', '2-methyl-2-propenyl isobutyrate', 'a-trans-b-bergamotene', 'E-ocimene epoxide', 'ethanone-1,1(1,4-phenylone)-bis', 'Isopauthenol', '10-Hydroxy-a-gurjunene', '5-methyl-5-ethenyl-2-hydroxytetrahydrofuran', 'Pentanoic acid, eugenyl ester', 'Nor-copaanone', 'E-isolimonene', 'E-dihydrocarveol', '2-acetyl-dihydro-2(H)-furanone', '6-methyl-5-Hepten-2-acetate', 'p-1(7),3,8-menthatriene', '3,7-dimethyl-1-octenylcyclopropanol', '2-methylbutenal', 'cis-Geranyl acetone', 'E-2-decenal', 'Z-g-cadinene', 'Cyclic-b-Ionone', "4-Methyl-6-(2’-methyl-1'-propenyl)-3,6-dihydro-2H-pyran", 'b-Safranyl acetate', 'Epoxy rose furan', 'Octan-1,5-olide', '1,7-dimethyl-4-isopropyl-2,7-cyclodecadien-1-ol', 'Formyl-cyclohenexe', 'Preziza-7(15)-en-3a-ol', 'N-formyl-leucine methyl ester', '2-Hydroxy 5-methoxy p-cymene', '1,10-epi-cubenol', 'p-menth-6-en-3,8-diol', 'trans-alpha bergamotene', 'E-p-sesquiphellandrol', '10-isobutyryloxy-8,9-didehydrothymol isobutyrate', 'E,E-4,5-Dibutyl-2,2,7,7-tetramethyl-3,5-octadiene', '1,4-dihydro-2-methyl-2H-3,1-benzoxazine', 'Himachaleneepoxide', 'Decan-1,5-olide', 'Helioropin', 'E-a-Ionene', 'E-1 -propenyl propyl disulfide', 'Benzene,1,1’-(1,5-hexadiene-1,6-diyl)bis', 'Methyl heneicosanate', 'a-fenchyl acetate', 'Lilac alcohol C', '4,7(11)-selinadiene', 'p-menth-1-ene-9-carboxaldehyde', '2-methyl-5(1-methylethenyl)-2-cyclohexenyl acetate', 'methyl 9,12,15-0ctadecatrienoate', 'pinocamphene', 'Methyl-11,14-eicosadienoate', 'Methyl (2R)-2-acetoxy-3-methylbutanoate', 'Z-7-decen-1,4-olide', '4-acetyl-1-methylcyclohexane', 'Z-muurol-5-en-4-ol', 'mentha-2,8-dienes', 'p-cymen-8-ol', 'tetramethyl benzene 2', 'Methoxy-p-tolyl-2-propanol', 'E-isocitral', 'Hexadecanoicacid,cyclohexylester', 'menthatriene isomer', '7-isobutyryloxythymol methyl ether', 'Z-isovalancenal', 'p-menth-1-en-8(9)-epoxide', 'dihydro-a-santalol', 'Acetocyclohexane dione (2)', 'Z-b-Farnesene', 'b–dihydroagarofuran', '2-butenoic acid, 3-methyl, ethyl ester', 'Z,Z,Z-Hexadeca-7,10,13-trienal', 'methyl-2-hydroxyisovalerate', 'Phenylacetoaldoxime (syn)', 'Phenylacetaldoxime O-methyl ether', 'Isovaleraldoxime', 'cyclosantalal', 'Methyl napthalene', 'nonanaI', '15-Acetoxylabdan-8-ol', '2-butyric acid, 3-methyl-, methyl ester ', 'E-1 -propenyl propyl tetrasulfide', 'cis-p-2,8-menthadien-1-ol', 'E,Z-2,6,10-trimethyl-2,6,10,12-tridecatetraene', 'Benzyl alcohol, a,a-dimethyl', 'p-1(7),2,8-menthatriene', 'kaurene-16-ol', 'cis-Cadinene ether', '4-hydroxy-8,9-didehydrothymol dimethyl ether', 'fenchyl valerate', '5.9-10-Kaur-15-ene', 'Dehydrogeosmin', 'diidroedulan', '2-phenethyl benzoate', '2,2-dimethyl-7-methoxy-6-vinylchromene', 'Ethyl 3-methylbutanoate', '6-phenyl-2,4-hexadiyne', 'Methyl-2-methyl butanoate', 'occidentalol acetate', 'Propyl decylbenzene', 'E-hydroxylinalool', 'Cumin', 'Decan-1,4-olide', 'E,E-a-farnesol', 'Fenchene', 'Methyl-2-hydroxy-3-methylvalerate', 'E,E-a-ionone', '2-Methyl hexylbutanoate', 'Conophthorine', 'b-lonone', 'Z-1 -propenyl propyl tetrasulfide', 'cis-limonene oxide', 'Dodecenylsuccinicanhydride', 'Octan-1,4-olide', 'Z-hexenyl acetate', 'E-1 -propenyl methyl trisulfide', 'E-3(4)-Epoxy-3,7-dimethylocta-1,6-diene', 'E-2,6-dimethyl-10-(p-tolyl)-undeca-2,6-diene', 'Isoleucic acid methyl ester', '4-methyl-5-hexen-1,4-olide', 'humulene epoxide III', 'caryophylla-2(12),6(13)-dien-5b-ol', '3,3,8,8-Tetramethyl-tricyclo[5.1.0.0(2,4)] oct-5-ene-5-propanoic acid', 'Acora-2,4 (15)-diene', '1,5-pentadiyl acetate', '2-pentyl octanoate', '5,6,7,7a-Tetrahydro-4, 4,7 a-trimethyl-2 (4 H)-Benzofuranone', 'Z-allo-ocimene', 'santolinatriene', 'Z-p-menthan-2-ol-1,8-oxide', 'b-phenylethyl n-butyrate', 'p-menth-5-en-2-one', 'epi-b-santalol', 'Z,Z-Hexadeca-7,10-dienal', '10-isovaleroxy-8,9-didehydrothymol isobutyrate', '7aH,10bH-Cadina-1(6),4-diene', '10-isovaleroxy-8,9-didehydrothymol methyl ether', 'Phenylacetoaldoxime (ant)', 'Butanoic acid, 2-methyl-2methyl butyl', 'tetramethyl benzene 1', 'b-Selenine', 'spirosantalo', 'E-2-undecanal', 'Mint sulphide type', 'Bicyclo[2.2.1 ]hept-2-ene,dimethyl', 'Tetradecan-1,5-olide', 'Dihydrofarnesyl acetone', '7-aH-silphiperphol-5-ene', '9-(2-methylbutyryloxy) thymol isobutyrate', 'Ethyl octylbenzene', 'Methyl 2-methyl-2(E)-butenoate', 'isopinocamphene', '5-hydroxy-p-mentha-6-ene-2-one', 'pinocarveol A', 'Dimethylstyrene', '8-Oxo-Neoisolongifolene', 'Methyl decylbenzene', 'E-b-ocimene-6,7-epoxide', 'Pent-4-en-1-yl isothiocyanate', 'Z-Methyl hexenoate', 'E,E-4,8,12-Trimethyltrideca-1,3,7,11-tetraene ', '7(11)-Epoxi-megastigma-5(6)-en-9-one', 'Octen-4-ol', '4-methylpent-2-en-4-olide', 'd14 cymene', 'Pentyl octylbenzene', 'Methyl octacosanate', '8,9-didehydrothymol isobutyrate', 'Stragol', 'Iso-Z-calamene', 'E,Z-Octa-3,5-dien-2-one', 'E-epoxy-Ocimene', '2,7-dimethyl-2,6-octadien-2-ol', 'p-menth-1-en-4(8)-epoxide', '9-isobutyryloxy thymol isobutyrate', 'Longipinanol', 'neoisodihydrocarvyl acetate', 'Z-b-sesquiphellandrol', 'Benzenedicarboxylicacidderivative', '2-pentyl pentanoate', 'Z-4-hexenyl acetic acid', 'Cycloisosativene', '3,5-dimethyl-2,3-dihydroxy-4H-pyran-4-one', '2-methoxy-p-tolyl-1-propanol', 'Z-1 -propenyl methyl trisulfide', '6-methoxy-8,9-didehydrothymol isobutyrate', 'totarene', '3-hydroxy-2-butanyl butyrate', 'Hexatrienoic acid methyl ester', '7-b-H-silphiperfol-5-ene', 'Geranyl ester II', 'Butyl nonylbenzene', 'epi-cyclosantalal', '9-isovaleroxythymol isobutyrate', 'T-muurolol', 'Z,Z,Z-Heptadeca-8,11,14-trienal', 'a-menthadien-8-ol-isomer', 'dimethylfuranlactone', 'Phytol isomer', 'Z-2-methyl-5-isopropenyl-2-vinyltetrahydrofuran', '9-isovaleroxythymol methyl ether', 'E-decenal', 'Methylcyclohex-1-en-4-one', 'Phenylethyl acohol', 'propyl 2-propenyl tetrasulfide', 'Pentyn-1-ol,4-methyI', 'Hex-5-ene nitrile', 'Caryophylla-3(15),7-dienol(6) II', 'Cadina-1,3,5-triene', '3-ethyl-1,4-pentadiene', '5-hydroxycineole', 'a-bergamotal', 'dimethylquinoline', '3-methylthioallylnitrile', '2-methoxy-4-vinyl-phenyl', 'Methyl Z3,Z6,E8-dodecatrien-1-ol', '3-(2-Isopropyl-5-methylphenyl)-2- methylpropionic acid', '(8b,13b)-kaur-16-ene', 'p-mentadiene n.i.', 'Z-m-mentha-2,8-diene', 'E-(cis-Carveol) epoxide', '4 Methylenecyclohexene', 'Thujene', 'Elsholtzia oxide', 'a- cardinal', 'E-Farnesyl acetate', 'Linalool oxide (furanoid) 2', 'Z-9-methyl octadecanoate', '2,3-dimethyl-5-heptenal', 'a-fenchene', 'Linalool oxide I', '2,3-dihydro-2-octyl-5-methylfuran-3-one', 'E-2(3)-Epoxy-2,6-dimethylnona-6,8-diene', '6-methyl-5- hepten-3-one', '9-isobutyryloxythymol methyl ether', 'Methyl (2R,3S)-2-acetoxy-3-methylpentanoate', '1-allyl-2,4-dimethoxybenzene', '3-hydroxy-4,4,dimethyl 2(3H)-furanone', 'Z-b-Farnesol', '7-Epi a-Eudesmol', 'caryophylla-2(12),6-dien-5b-ol', 'trans-allo ocimene', 'Methyl-2-methyl propanoate', '2,7-dimethyl-octa-3,5-diene', '1,2,4,4-tetramethylcyclopentane', 'Z,E-4,8,12-Dimethyltrideca-1,3,7,11-tetraene', 'Pinene', 'N-formyl-isoleucine methyl ester', 'b–acoradiene', 'Butyl hexylbenzene', '5-epi-7-epi-a-eudesmol', '5-hydroxy-2-furancarboxyaldehyde', 'Z,Z-1,5-cyclodecadiene', '3-methylen-7,11-dimethyl-dodeca-1,6,10-triene', '3-Methyl butylbutanoate', 'isovaleraldehyde oxime (syn)', 'Z-3-Hexenol propanoate', 'Hexan-1,4-olide', 'Linalool oxide(pyranoid; ketone)', '4bH,10aH-Guaia-1(5),6-diene', '1,4-benzene dicarboxaldehyde', 'prenyl ethanoate', '2-methanol-bicyclo[3.1.1]hept-2-ene', 'trans-linalool-6,7-epoxide', 'Lilac alcohol B', 'trans-pinocamphone', '6-Isopropenyl-4,8a-dimethyl-1,2,3,5,6,7,8,8a_x0002_Octahydro-naphthalen-2-ol', 'Z -limonene oxide', 'Z-g-caryophyllene', 'Z-b-ocimene-6,7-epoxide', 'Ethyl (E)-3,7-dimethyl-3,6-octadienoate', 'E-Cembrene A', 'E-2,6-Dimethyl-3,5,7-octatriene-2-ol', 'Linalool oxide (pyranoid) 2', 'Anthranilic aldehyde', 'eudesmol', 'E,E-heptadienol', 'd-carene', '4-epiabietal', 'E,Z-decadienal', 'Glycerylpalmitate', 'E-b-farnesol', 'E,E-4,8,12-Dimethyltrideca-1,3,7,11-tetraene', 'C10 H14 alkylbenzene', 'bis-(2-Ethylhexyl)1,2-benzene dicarboxylicacidester', 'khusinol', '4-methylcyclopentan-1,2-dione', 'Z,Z-Nonan-3,6-dien-1-ol', '10-(2-methylbutyryloxy)-8,9-didehydrothymol methyl ether', '3,7,11-trimethyl-dodeca-1,6,10-triene', 'germacrene D', 'nerol-1,5-oxide', '3,5-dimethyl-tetrahydro-2H-pyran-2-one', 'cis-linalool-6,7-epoxide', '3-methylene-7-methyl-1,6-octadien-4-ol', 'N-benzylidene-isoleucine methyl ester', 'Nerolidyl methyl ether', '2,7,7-Trimethylbicyclo-[3.1.1]hept-2-en-6-ol acetate', 'Dihydro linalool', 'E,E-geranyl linalool', '3-cyclohexenyl-prop-2-enal', 'Propyl heptyl benzene', 'Z-a-Bisabolene epoxide', '6-(2-butyl)quinoline', '4aH,10aH-Guaia-1(5),6-diene', '7-isobutyryloxy thymol isobutyrate', 'bicyclo(4,1,0)heptan-5-ol,trans, (+)-trans-3,7,7-trimethyl-', 'a-Terpinene-4-ol', 'Z,Z-hentriacontadiene', '3,9-epoxy-p-menth-1-ene', 'Z-3-hexenyl isobutanoate', 'Calamenene(1-S,cis)', '1,10-di-epi-cubenol', 'acetyldihydroalbene (15)', 'E-carvone epoxide', 'alkyl tiglate', '5,6 Epoxy.3.3.6-trimethyI-1-hepten-4-on', 'humulene epoxide – I', 'Z,Z-nonacosadiene', 't-Eudesm-4(15),7-dien-12-ol', 'E,E-2,6,10-trimethyl-2,6,10,12-tridecatetraene', 'Isobutoxyl-2-propyl-cyclopropane', 'Methylphenol', 'N-benzylidene-leucine methyl ester', 'Aacetoine acetate', 'Methyl tricosanate', 'Propyl octylbenzene', 'Methyl (E)-3,7-dimethyl-3,6-octadienoate', 'Methyl hexacosanate', '2-methyl-hexyl butanoate', 'Lilac alcohol A', 'Methyl propyl phenol', 'tert-muurolol', '2-cyclohexen-1-one, 4-hydroxy-3-methyl-6-(1-methylethyl), (cis or trans)', 'dihydro carveole', 'Linalool oxide (furanoid) 1', '2-Methyl- 2-bornene', 'methyl isothymol', 'Geranyl ester I', 'caryophylla-2(12),6(13)-dien-5a-ol', 'a,p-dimethyl styrene', 'iso-dihydrocarveol', 'cis-14-nor-Muurol-5-en-4-one', 'epi-vitispirane', '2-methylbutyraldehyde oxime (syn)', 'Nor-bourbonene', 'p-mentha-2,4(8),6-trien-2,3-diol', '8-a-11-elemenediol', 'Linalool oxide (pyranoid) 1', 'Elsloltziadiol-2-acetate', '7-isobutyryloxy-8,9-didehydrothymol isobutyrate', 'Elshotziadiol', 'E-(trans-Carveol) epoxide', 'Limonene oxide (isomer)', '1,4-dimethyl-3-cycloexen-1-ol-1-ethanone', 'Methyl-13-methyl pentadecanoate', '2,4-dimethylphenylethanone', 'E-1 -propenyl propyl trisulfide', '10-isobutyryloxy-8,9-didehydro thymol methyl ether', 'Z-thujopsadiene', 'dihydrodemethoxyencecalin', '6,7-epoxylinalool', '2-methyl-6-methylene-1,7-octadiene-3-ol', 'eudesma-4(15),7-dien-1b-ol', 'caryophyllene alcohol', 'Cedrene-8,13-ol acetate', 'Z3,Z6,E8-dodecatrien-1-ol', 'Z-hydroxylinalool', 'ligul oxide', '1,5,9,9-Tetramethyl-2-methylenespiro[3.5]non-5-ene', 'Z,Z-9,12-methyl octadecadienoate', 'E-1-propenyl propyl pentasulfide', 'Heptatriene-1,3,6-trimethyl', 'diisobutylphathalate', 'alkyl angelate', 'E,E-b-ionone', 'Methyl geranylate', '(1-methylethyenyl)benzene', 'methyl 4-methoxy-salicylate', 'Butyl-2-ethylhexylphthalate', 'trans-dihydro-a-terpineol', '1(10),4,11-Germacratrien-9-ol', 'Z-1 -propenyl propyl pentasulfide', 'theaspirane (2)', '9,10-Dehydroisolongifolene', 'methyl thymol', '9-(2-methylbutyryloxy)thymol methyl ether', '2-methyl-6-methylene-octa-1,7-diene-3-one', 'Pentyl hexyl benzene', '9-cedranone', '2-phenylethyl methylether', 'selin-11-en-4a-ol', 'labda–8,14–dien–13–ol', '2,6-dimethyl-3,7-octadiene-2,6-diol', '2,9-octadecanoic acid', 'a-camphenal', '9-octadecenol', 'a-fenchol', 'Z-4-methyldecano-5-lactone', '2,5-dimethyl-3-(3-methylbutyl)-pyrazine ', 'p-mentha-1,8-diene-4-ol', '2,6,6-trimethyl-2-vinyl-THP-5-one']
            present_compounds = list(filter(lambda x: x not in missing_compound, present_compounds))
            compound_fingerprints = data2_indexed.loc[present_compounds]
            
        if not present_compounds:
            aggregated_profile = np.zeros(data2_indexed.shape[1])
            new_feature_list.append(aggregated_profile)
            continue
        
        # Aggregate based on the chosen method
        if agg_method == 'mean':
            aggregated_profile = compound_fingerprints.mean(axis=0).values
        elif agg_method == 'sum':
            aggregated_profile = compound_fingerprints.sum(axis=0).values
        elif agg_method == 'binary':
            aggregated_profile = (compound_fingerprints.sum(axis=0) > 0).astype(int).values
        else:
            raise ValueError("agg_method must be 'mean', 'sum', or 'binary'")
            
        new_feature_list.append(aggregated_profile)

    
    # Create the new feature DataFrame
    # X_aggregated = pd.DataFrame(new_feature_list, columns=[count]+X_original.columns.tolist()+data2_indexed.columns.tolist())
    X_aggregated = pd.DataFrame(new_feature_list, columns=data2_indexed.columns)
    
    # Evaluate the model with these new features
    X_train, X_test, y_train, y_test = train_test_split(X_aggregated, y, test_size=0.33, random_state=42)
    columns_with_nan = X_train.isna().any()
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Generated feature matrix shape: {X_aggregated.shape}")
    print(f"Model accuracy with aggregated features: {acc:.4f}\n")
    return X_aggregated, acc


def strategy_2_weighted_profile(data1, data2, data2_indexed):
    """
    Strategy 2: Creates a fingerprint profile weighted by compound importance.
    """
    print("--- Running Strategy 2: Weighted Aggregated Profile ---")
    
    # Step 1: Train a preliminary model on original data to get feature importances
    prelim_model = RandomForestClassifier(random_state=42)
    prelim_model.fit(X_original, y)
    importances = pd.Series(prelim_model.feature_importances_, index=X_original.columns)
    
    # Step 2: Create weighted profiles
    new_feature_list = []
    for index, row in X_original.iterrows():
        present_compounds = row[row == 1].index.tolist()
        
        if not present_compounds:
            new_feature_list.append(np.zeros(data2_indexed.shape[1]))
            continue
        
        try:
            try:
                present_compounds[present_compounds.index('6-methyl-5- hepten-2-one')] = '6-methyl-5-hepten-2-one'
            except:pass
            compound_fingerprints = data2_indexed.loc[present_compounds]
        except:
            missing_compound = missing_compound = missing_compound = ['Z-3-Hexenyl-a-methyl butyrate', 'dodecan-1,5-olide', 'Azuleno', '2,3-dimethyl-5-heptanal', 'Methyl nerylate', '10-(2-methylbutyryloxy)-8,9-didehydrothymol isobutyrate', '8,9-didehydrothymol methyl ether', 'cis-3-hexenyl tiglate', 'l-pentadecene', 'dehydrosabina ketone', 'Z-pinene hydrate', 'trans-g-cadinene', 'C15H24O', 'Heptane-1,5-olide', 'Methyl tetracosanate', 'p-cumene-8-ol', 'a-oxo-ethanol-benzoate', 'E-g-bisabolene', '9,10-Dihydroisolongifolene', 'cis-p-mentha-1(7),8-dien-2-ol', 'E-allo ocimene', 'E-p-menthan-2-ol-1,8-oxide', 'Lilac alcohol D', 'Methyl 2-methyl-2(Z)-butenoate', 'E-a-farnesene', 'Sylvestrane', '2-methyl-6-methylene-1,7-octadiene-2-one', 'Butyl octylbenzene', '5-methyl-2-hexanal', 'Linalool oxide(pyranoid; alcohol)', '8,8-dimethyl-4-methylene-1-oxoaspiro(2,5)-oct-5-ene', 'isobutyraldehyde oxime (syn)', '5-methyl-6-en-2-one', 'E-ene-yne-Dicycloether', 'Dimethyl disulphide monosulphone', 'Dehydro sabine ketone', 'mesistylene', '4a-a,7-b,7a-a-nepetalactone', 'E-2,6-DimethyInona-6,8-dien-4-ol', 'methyl benzoate. 2.5-dimethyl-', 'Z,Z-Heptadeca-8,11-dienal', 'C15H28o', '3-Methyl isobutylbutanoate', 'Methyl docosanate', 'Methyl eicosanate', '5-phenylmethoxy-1-pentanol', 'cis-limonene epoxide', 'cis-Isopulegone', '2-hydroxy piperitone', '14-Hydroxy-Z-caryophyllene', '6,7-dihydro-8-hydroxylinalool', '4-hydroxy-2-methylcyclopentan-1,3-dione', 'cis-linalool oxide acetate (pyranoid)', 'caryophylla-2(12),6-dien-5a-ol', 'Propyl nonyl benzene', 'a-p-Dimethyl styrene', '3,5-dimethyl-pyran-4-one', 'isocyanato-methylbenzene', 'Xanthorrhizol isomer', 'E-(E)-7-Mega stigmen-3,9-dione', 'trans-p-mentha-1(7),8-dien-2-ol', 'E-Geranyl acetone', 'ar-Curcumene 15-al', 'E-8-undecanal', 'Dehydroelshotzia ketone', '3,5-dimethyl methyl benzoate', 'Decahydro-4,4,8,9,10-pentamethylnaphthalene', 'pinocarveol B', '14–hydroxy–d–cadinene', ' 7,13-abietadiene', 'Pentyl-2-methyl-2-butanoate', 'Pentyl heptylbenzene', 'labda-7,14-dien-13-ol', '3-methylamyl valerate', 'abieta-7,13-diene-3-one', 'Methyl-4-methoxycyclohex-1-ene', 'cis-tent-muurola-3,5-dien', 'E-2,6-DimethyInona-3,6,8-trien-2-ol', '2E,4E-nonadienal', 'E-b-ocimene epoxide', '4,8 b-Epoxycaryophyllene', 'Z-5-Tricosene', 'p-cymen-8-yl acetate', '4-methyl pentanone', '2-pentyl propanoate', '3-methylcyclopentan-1,2-dione', 't,4-dimethylbenzene butanal', 'Methyl ester hexadecanoic acid', 'cis-a-bergamotol', '9-19-Cycloanost_x0002_6-ene-3,7-diol', 'Z-cinnamic alcohol', 'Methyl (2R)-2-acetoxy-4-methylpentanoate', 'b-Safranol', '2,2-dimethoxy propanone', 'Methyl (2S,3S)-2-acetoxy-3-methylpentanoate', '8-Acetoxypatchoulialcohol', 'Hexadecenoic acid', '3-Methyl butylhexanoate', 'Dimethyl napthalene', '2-methyl-2-propenyl isobutyrate', 'a-trans-b-bergamotene', 'E-ocimene epoxide', 'ethanone-1,1(1,4-phenylone)-bis', 'Isopauthenol', '10-Hydroxy-a-gurjunene', '5-methyl-5-ethenyl-2-hydroxytetrahydrofuran', 'Pentanoic acid, eugenyl ester', 'Nor-copaanone', 'E-isolimonene', 'E-dihydrocarveol', '2-acetyl-dihydro-2(H)-furanone', '6-methyl-5-Hepten-2-acetate', 'p-1(7),3,8-menthatriene', '3,7-dimethyl-1-octenylcyclopropanol', '2-methylbutenal', 'cis-Geranyl acetone', 'E-2-decenal', 'Z-g-cadinene', 'Cyclic-b-Ionone', "4-Methyl-6-(2’-methyl-1'-propenyl)-3,6-dihydro-2H-pyran", 'b-Safranyl acetate', 'Epoxy rose furan', 'Octan-1,5-olide', '1,7-dimethyl-4-isopropyl-2,7-cyclodecadien-1-ol', 'Formyl-cyclohenexe', 'Preziza-7(15)-en-3a-ol', 'N-formyl-leucine methyl ester', '2-Hydroxy 5-methoxy p-cymene', '1,10-epi-cubenol', 'p-menth-6-en-3,8-diol', 'trans-alpha bergamotene', 'E-p-sesquiphellandrol', '10-isobutyryloxy-8,9-didehydrothymol isobutyrate', 'E,E-4,5-Dibutyl-2,2,7,7-tetramethyl-3,5-octadiene', '1,4-dihydro-2-methyl-2H-3,1-benzoxazine', 'Himachaleneepoxide', 'Decan-1,5-olide', 'Helioropin', 'E-a-Ionene', 'E-1 -propenyl propyl disulfide', 'Benzene,1,1’-(1,5-hexadiene-1,6-diyl)bis', 'Methyl heneicosanate', 'a-fenchyl acetate', 'Lilac alcohol C', '4,7(11)-selinadiene', 'p-menth-1-ene-9-carboxaldehyde', '2-methyl-5(1-methylethenyl)-2-cyclohexenyl acetate', 'methyl 9,12,15-0ctadecatrienoate', 'pinocamphene', 'Methyl-11,14-eicosadienoate', 'Methyl (2R)-2-acetoxy-3-methylbutanoate', 'Z-7-decen-1,4-olide', '4-acetyl-1-methylcyclohexane', 'Z-muurol-5-en-4-ol', 'mentha-2,8-dienes', 'p-cymen-8-ol', 'tetramethyl benzene 2', 'Methoxy-p-tolyl-2-propanol', 'E-isocitral', 'Hexadecanoicacid,cyclohexylester', 'menthatriene isomer', '7-isobutyryloxythymol methyl ether', 'Z-isovalancenal', 'p-menth-1-en-8(9)-epoxide', 'dihydro-a-santalol', 'Acetocyclohexane dione (2)', 'Z-b-Farnesene', 'b–dihydroagarofuran', '2-butenoic acid, 3-methyl, ethyl ester', 'Z,Z,Z-Hexadeca-7,10,13-trienal', 'methyl-2-hydroxyisovalerate', 'Phenylacetoaldoxime (syn)', 'Phenylacetaldoxime O-methyl ether', 'Isovaleraldoxime', 'cyclosantalal', 'Methyl napthalene', 'nonanaI', '15-Acetoxylabdan-8-ol', '2-butyric acid, 3-methyl-, methyl ester ', 'E-1 -propenyl propyl tetrasulfide', 'cis-p-2,8-menthadien-1-ol', 'E,Z-2,6,10-trimethyl-2,6,10,12-tridecatetraene', 'Benzyl alcohol, a,a-dimethyl', 'p-1(7),2,8-menthatriene', 'kaurene-16-ol', 'cis-Cadinene ether', '4-hydroxy-8,9-didehydrothymol dimethyl ether', 'fenchyl valerate', '5.9-10-Kaur-15-ene', 'Dehydrogeosmin', 'diidroedulan', '2-phenethyl benzoate', '2,2-dimethyl-7-methoxy-6-vinylchromene', 'Ethyl 3-methylbutanoate', '6-phenyl-2,4-hexadiyne', 'Methyl-2-methyl butanoate', 'occidentalol acetate', 'Propyl decylbenzene', 'E-hydroxylinalool', 'Cumin', 'Decan-1,4-olide', 'E,E-a-farnesol', 'Fenchene', 'Methyl-2-hydroxy-3-methylvalerate', 'E,E-a-ionone', '2-Methyl hexylbutanoate', 'Conophthorine', 'b-lonone', 'Z-1 -propenyl propyl tetrasulfide', 'cis-limonene oxide', 'Dodecenylsuccinicanhydride', 'Octan-1,4-olide', 'Z-hexenyl acetate', 'E-1 -propenyl methyl trisulfide', 'E-3(4)-Epoxy-3,7-dimethylocta-1,6-diene', 'E-2,6-dimethyl-10-(p-tolyl)-undeca-2,6-diene', 'Isoleucic acid methyl ester', '4-methyl-5-hexen-1,4-olide', 'humulene epoxide III', 'caryophylla-2(12),6(13)-dien-5b-ol', '3,3,8,8-Tetramethyl-tricyclo[5.1.0.0(2,4)] oct-5-ene-5-propanoic acid', 'Acora-2,4 (15)-diene', '1,5-pentadiyl acetate', '2-pentyl octanoate', '5,6,7,7a-Tetrahydro-4, 4,7 a-trimethyl-2 (4 H)-Benzofuranone', 'Z-allo-ocimene', 'santolinatriene', 'Z-p-menthan-2-ol-1,8-oxide', 'b-phenylethyl n-butyrate', 'p-menth-5-en-2-one', 'epi-b-santalol', 'Z,Z-Hexadeca-7,10-dienal', '10-isovaleroxy-8,9-didehydrothymol isobutyrate', '7aH,10bH-Cadina-1(6),4-diene', '10-isovaleroxy-8,9-didehydrothymol methyl ether', 'Phenylacetoaldoxime (ant)', 'Butanoic acid, 2-methyl-2methyl butyl', 'tetramethyl benzene 1', 'b-Selenine', 'spirosantalo', 'E-2-undecanal', 'Mint sulphide type', 'Bicyclo[2.2.1 ]hept-2-ene,dimethyl', 'Tetradecan-1,5-olide', 'Dihydrofarnesyl acetone', '7-aH-silphiperphol-5-ene', '9-(2-methylbutyryloxy) thymol isobutyrate', 'Ethyl octylbenzene', 'Methyl 2-methyl-2(E)-butenoate', 'isopinocamphene', '5-hydroxy-p-mentha-6-ene-2-one', 'pinocarveol A', 'Dimethylstyrene', '8-Oxo-Neoisolongifolene', 'Methyl decylbenzene', 'E-b-ocimene-6,7-epoxide', 'Pent-4-en-1-yl isothiocyanate', 'Z-Methyl hexenoate', 'E,E-4,8,12-Trimethyltrideca-1,3,7,11-tetraene ', '7(11)-Epoxi-megastigma-5(6)-en-9-one', 'Octen-4-ol', '4-methylpent-2-en-4-olide', 'd14 cymene', 'Pentyl octylbenzene', 'Methyl octacosanate', '8,9-didehydrothymol isobutyrate', 'Stragol', 'Iso-Z-calamene', 'E,Z-Octa-3,5-dien-2-one', 'E-epoxy-Ocimene', '2,7-dimethyl-2,6-octadien-2-ol', 'p-menth-1-en-4(8)-epoxide', '9-isobutyryloxy thymol isobutyrate', 'Longipinanol', 'neoisodihydrocarvyl acetate', 'Z-b-sesquiphellandrol', 'Benzenedicarboxylicacidderivative', '2-pentyl pentanoate', 'Z-4-hexenyl acetic acid', 'Cycloisosativene', '3,5-dimethyl-2,3-dihydroxy-4H-pyran-4-one', '2-methoxy-p-tolyl-1-propanol', 'Z-1 -propenyl methyl trisulfide', '6-methoxy-8,9-didehydrothymol isobutyrate', 'totarene', '3-hydroxy-2-butanyl butyrate', 'Hexatrienoic acid methyl ester', '7-b-H-silphiperfol-5-ene', 'Geranyl ester II', 'Butyl nonylbenzene', 'epi-cyclosantalal', '9-isovaleroxythymol isobutyrate', 'T-muurolol', 'Z,Z,Z-Heptadeca-8,11,14-trienal', 'a-menthadien-8-ol-isomer', 'dimethylfuranlactone', 'Phytol isomer', 'Z-2-methyl-5-isopropenyl-2-vinyltetrahydrofuran', '9-isovaleroxythymol methyl ether', 'E-decenal', 'Methylcyclohex-1-en-4-one', 'Phenylethyl acohol', 'propyl 2-propenyl tetrasulfide', 'Pentyn-1-ol,4-methyI', 'Hex-5-ene nitrile', 'Caryophylla-3(15),7-dienol(6) II', 'Cadina-1,3,5-triene', '3-ethyl-1,4-pentadiene', '5-hydroxycineole', 'a-bergamotal', 'dimethylquinoline', '3-methylthioallylnitrile', '2-methoxy-4-vinyl-phenyl', 'Methyl Z3,Z6,E8-dodecatrien-1-ol', '3-(2-Isopropyl-5-methylphenyl)-2- methylpropionic acid', '(8b,13b)-kaur-16-ene', 'p-mentadiene n.i.', 'Z-m-mentha-2,8-diene', 'E-(cis-Carveol) epoxide', '4 Methylenecyclohexene', 'Thujene', 'Elsholtzia oxide', 'a- cardinal', 'E-Farnesyl acetate', 'Linalool oxide (furanoid) 2', 'Z-9-methyl octadecanoate', '2,3-dimethyl-5-heptenal', 'a-fenchene', 'Linalool oxide I', '2,3-dihydro-2-octyl-5-methylfuran-3-one', 'E-2(3)-Epoxy-2,6-dimethylnona-6,8-diene', '6-methyl-5- hepten-3-one', '9-isobutyryloxythymol methyl ether', 'Methyl (2R,3S)-2-acetoxy-3-methylpentanoate', '1-allyl-2,4-dimethoxybenzene', '3-hydroxy-4,4,dimethyl 2(3H)-furanone', 'Z-b-Farnesol', '7-Epi a-Eudesmol', 'caryophylla-2(12),6-dien-5b-ol', 'trans-allo ocimene', 'Methyl-2-methyl propanoate', '2,7-dimethyl-octa-3,5-diene', '1,2,4,4-tetramethylcyclopentane', 'Z,E-4,8,12-Dimethyltrideca-1,3,7,11-tetraene', 'Pinene', 'N-formyl-isoleucine methyl ester', 'b–acoradiene', 'Butyl hexylbenzene', '5-epi-7-epi-a-eudesmol', '5-hydroxy-2-furancarboxyaldehyde', 'Z,Z-1,5-cyclodecadiene', '3-methylen-7,11-dimethyl-dodeca-1,6,10-triene', '3-Methyl butylbutanoate', 'isovaleraldehyde oxime (syn)', 'Z-3-Hexenol propanoate', 'Hexan-1,4-olide', 'Linalool oxide(pyranoid; ketone)', '4bH,10aH-Guaia-1(5),6-diene', '1,4-benzene dicarboxaldehyde', 'prenyl ethanoate', '2-methanol-bicyclo[3.1.1]hept-2-ene', 'trans-linalool-6,7-epoxide', 'Lilac alcohol B', 'trans-pinocamphone', '6-Isopropenyl-4,8a-dimethyl-1,2,3,5,6,7,8,8a_x0002_Octahydro-naphthalen-2-ol', 'Z -limonene oxide', 'Z-g-caryophyllene', 'Z-b-ocimene-6,7-epoxide', 'Ethyl (E)-3,7-dimethyl-3,6-octadienoate', 'E-Cembrene A', 'E-2,6-Dimethyl-3,5,7-octatriene-2-ol', 'Linalool oxide (pyranoid) 2', 'Anthranilic aldehyde', 'eudesmol', 'E,E-heptadienol', 'd-carene', '4-epiabietal', 'E,Z-decadienal', 'Glycerylpalmitate', 'E-b-farnesol', 'E,E-4,8,12-Dimethyltrideca-1,3,7,11-tetraene', 'C10 H14 alkylbenzene', 'bis-(2-Ethylhexyl)1,2-benzene dicarboxylicacidester', 'khusinol', '4-methylcyclopentan-1,2-dione', 'Z,Z-Nonan-3,6-dien-1-ol', '10-(2-methylbutyryloxy)-8,9-didehydrothymol methyl ether', '3,7,11-trimethyl-dodeca-1,6,10-triene', 'germacrene D', 'nerol-1,5-oxide', '3,5-dimethyl-tetrahydro-2H-pyran-2-one', 'cis-linalool-6,7-epoxide', '3-methylene-7-methyl-1,6-octadien-4-ol', 'N-benzylidene-isoleucine methyl ester', 'Nerolidyl methyl ether', '2,7,7-Trimethylbicyclo-[3.1.1]hept-2-en-6-ol acetate', 'Dihydro linalool', 'E,E-geranyl linalool', '3-cyclohexenyl-prop-2-enal', 'Propyl heptyl benzene', 'Z-a-Bisabolene epoxide', '6-(2-butyl)quinoline', '4aH,10aH-Guaia-1(5),6-diene', '7-isobutyryloxy thymol isobutyrate', 'bicyclo(4,1,0)heptan-5-ol,trans, (+)-trans-3,7,7-trimethyl-', 'a-Terpinene-4-ol', 'Z,Z-hentriacontadiene', '3,9-epoxy-p-menth-1-ene', 'Z-3-hexenyl isobutanoate', 'Calamenene(1-S,cis)', '1,10-di-epi-cubenol', 'acetyldihydroalbene (15)', 'E-carvone epoxide', 'alkyl tiglate', '5,6 Epoxy.3.3.6-trimethyI-1-hepten-4-on', 'humulene epoxide – I', 'Z,Z-nonacosadiene', 't-Eudesm-4(15),7-dien-12-ol', 'E,E-2,6,10-trimethyl-2,6,10,12-tridecatetraene', 'Isobutoxyl-2-propyl-cyclopropane', 'Methylphenol', 'N-benzylidene-leucine methyl ester', 'Aacetoine acetate', 'Methyl tricosanate', 'Propyl octylbenzene', 'Methyl (E)-3,7-dimethyl-3,6-octadienoate', 'Methyl hexacosanate', '2-methyl-hexyl butanoate', 'Lilac alcohol A', 'Methyl propyl phenol', 'tert-muurolol', '2-cyclohexen-1-one, 4-hydroxy-3-methyl-6-(1-methylethyl), (cis or trans)', 'dihydro carveole', 'Linalool oxide (furanoid) 1', '2-Methyl- 2-bornene', 'methyl isothymol', 'Geranyl ester I', 'caryophylla-2(12),6(13)-dien-5a-ol', 'a,p-dimethyl styrene', 'iso-dihydrocarveol', 'cis-14-nor-Muurol-5-en-4-one', 'epi-vitispirane', '2-methylbutyraldehyde oxime (syn)', 'Nor-bourbonene', 'p-mentha-2,4(8),6-trien-2,3-diol', '8-a-11-elemenediol', 'Linalool oxide (pyranoid) 1', 'Elsloltziadiol-2-acetate', '7-isobutyryloxy-8,9-didehydrothymol isobutyrate', 'Elshotziadiol', 'E-(trans-Carveol) epoxide', 'Limonene oxide (isomer)', '1,4-dimethyl-3-cycloexen-1-ol-1-ethanone', 'Methyl-13-methyl pentadecanoate', '2,4-dimethylphenylethanone', 'E-1 -propenyl propyl trisulfide', '10-isobutyryloxy-8,9-didehydro thymol methyl ether', 'Z-thujopsadiene', 'dihydrodemethoxyencecalin', '6,7-epoxylinalool', '2-methyl-6-methylene-1,7-octadiene-3-ol', 'eudesma-4(15),7-dien-1b-ol', 'caryophyllene alcohol', 'Cedrene-8,13-ol acetate', 'Z3,Z6,E8-dodecatrien-1-ol', 'Z-hydroxylinalool', 'ligul oxide', '1,5,9,9-Tetramethyl-2-methylenespiro[3.5]non-5-ene', 'Z,Z-9,12-methyl octadecadienoate', 'E-1-propenyl propyl pentasulfide', 'Heptatriene-1,3,6-trimethyl', 'diisobutylphathalate', 'alkyl angelate', 'E,E-b-ionone', 'Methyl geranylate', '(1-methylethyenyl)benzene', 'methyl 4-methoxy-salicylate', 'Butyl-2-ethylhexylphthalate', 'trans-dihydro-a-terpineol', '1(10),4,11-Germacratrien-9-ol', 'Z-1 -propenyl propyl pentasulfide', 'theaspirane (2)', '9,10-Dehydroisolongifolene', 'methyl thymol', '9-(2-methylbutyryloxy)thymol methyl ether', '2-methyl-6-methylene-octa-1,7-diene-3-one', 'Pentyl hexyl benzene', '9-cedranone', '2-phenylethyl methylether', 'selin-11-en-4a-ol', 'labda–8,14–dien–13–ol', '2,6-dimethyl-3,7-octadiene-2,6-diol', '2,9-octadecanoic acid', 'a-camphenal', '9-octadecenol', 'a-fenchol', 'Z-4-methyldecano-5-lactone', '2,5-dimethyl-3-(3-methylbutyl)-pyrazine ', 'p-mentha-1,8-diene-4-ol', '2,6,6-trimethyl-2-vinyl-THP-5-one']
            present_compounds = list(filter(lambda x: x not in missing_compound, present_compounds))
            compound_fingerprints = data2_indexed.loc[present_compounds]
        
        if not present_compounds:
            aggregated_profile = np.zeros(data2_indexed.shape[1])
            new_feature_list.append(aggregated_profile)
            continue
        
        compound_fingerprints = data2_indexed.loc[present_compounds]
        
        if '6-methyl-5-hepten-2-one' in present_compounds:
            present_compounds[present_compounds.index('6-methyl-5-hepten-2-one')] = '6-methyl-5- hepten-2-one'
        compound_importances = importances.loc[present_compounds]
        
        
        # Calculate the weighted sum of fingerprints
        weighted_profile = compound_fingerprints.multiply(compound_importances, axis=0).sum(axis=0).values
        new_feature_list.append(weighted_profile)
        
    X_weighted = pd.DataFrame(new_feature_list, columns=data2_indexed.columns)
    
    # Evaluate a new model with these weighted features
    X_train, X_test, y_train, y_test = train_test_split(X_weighted, y, test_size=0.33, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Generated weighted feature matrix shape: {X_weighted.shape}")
    print(f"Model accuracy with weighted features: {acc:.4f}\n")
    return X_weighted, acc


def strategy_3_concatenated_features(data1, data2, data2_indexed):
    """
    Strategy 3: Concatenates original features with aggregated profiles.
    """
    print("--- Running Strategy 3: Concatenated Features ---")
    
    # Step 1: Generate the aggregated profile (re-using logic from Strategy 1)
    X_aggregated,acc = strategy_1_aggregated_profile(data1, data2, data2_indexed, agg_method='mean')
    
    # Step 2: Concatenate original and new feature sets
    # Use .reset_index to ensure proper alignment
    X_combined = pd.concat([X_original.reset_index(drop=True), X_aggregated.reset_index(drop=True)], axis=1)
    
    # Evaluate the model with the combined feature set
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.33, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("--- Results for Strategy 3 ---")
    print(f"Generated combined feature matrix shape: {X_combined.shape}")
    print(f"Model accuracy with combined features: {acc:.4f}\n")
    return X_combined, acc


# --- 3. MAIN EXECUTION ---
if __name__ == "__main__":
    fnamel = [ "fp_MACCS.csv", "fp_Pubchem.csv", "fp_EXT.csv", "fp_Sub_count.csv", "fp_Sub.csv", ]
    prefixl = ["MACCS", "Pubchem","EXT", "Sub-count", "Sub"]

    acc_data = pd.DataFrame()
    acc_data["index"] = ["base", "fingerprint mean", "fingerprint weighted mean", "original combined fingerprint mean"]
    for fname, prefix in zip(fnamel,prefixl):
        
        data2 = pd.read_csv(f"../data/{fname}")
        # data2 = data2.drop(["Name.1"],axis=1)
        # data2 = data2.drop(["Name.1", "Name.2", "Name.3"],axis=1)
        data2_indexed = data2.set_index('Name')
        
        # Run and evaluate each strategy
        acc_base = strategy_base()
        X_agg, acc1 = strategy_1_aggregated_profile(data1, data2, data2_indexed, agg_method='mean')
        # X_weighted, acc2 = strategy_2_weighted_profile(data1, data2, data2_indexed)
        X_combined, acc3 = strategy_3_concatenated_features(data1, data2, data2_indexed)
        
        acc_data[prefix] = [acc_base, acc1, 0 ,acc3] #[acc_base, acc1, acc2, acc3]

        data_with_fpmean = pd.concat([X_combined, y_all],axis = 1)
        data_with_fpmean.to_csv(f"output_cls_fp/data_with_fpmean_{prefix}.csv", index=False)
        
    acc_data.to_csv(f"output_cls_fp/accuracy.csv")   