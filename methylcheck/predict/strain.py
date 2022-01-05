import pandas as pd
import numpy as np
import methylcheck
from pathlib import Path
from scipy.stats import norm

try:
    from importlib import resources # py3.7+
except ImportError:
    import pkg_resources #py < 3.7
pkg_namespace = 'methylcheck.data_files'

import logging
LOGGER = logging.getLogger(__name__)


def mouse_beta_to_AF(beta_df, manifest=None):
    if not manifest:
        try:
            from methylprep import Manifest
            manifest = Manifest('mouse')
        except ImportError:
            raise ImportError("this function requires `methylprep` be installed (to read manifest array files), or to pass in a Manifest object.")
    #1 - get 'rs' probes from mouse manifest
    #rs_probes = manifest.data_frame[ manifest.data_frame.index.str.startswith('rs') ]
    rs_probes = manifest.snp_data_frame.set_index('IlmnID')
    #2 - extract RS probes, but make sure all of the probes exist
    rs_probes_found = set(list(rs_probes.index)) & set(list(beta_df.index))
    vafs = beta_df.loc[ rs_probes_found ]
    #3 - limit to only those that also have 'rs' listed in design string -- should be 1353 of 1486
    to_flip = rs_probes[ (rs_probes.design.str.contains('rs')) & (rs_probes.index.isin(rs_probes_found)) ].index
    #4 - and flip these betas
    flipped = (1- vafs.loc[ to_flip ])
    vafs.update(flipped)
    return vafs

def control_df_snps(filepath):
    """ loads a control_probes.pkl file and returns a dataframe with just the snp_beta portion for all samples"""
    raw = pd.read_pickle(Path(filepath))
    raw = {k: v['snp_beta'] for k,v in raw.items()}
    df = pd.DataFrame(data=raw)
    df = df.loc[ df.index.astype(str).str.startswith('rs') ]
    return df

def _dataframe_contains_snps(beta_df):
    # make sure it contains snps
    contains_snps_column = True if isinstance(beta_df, pd.DataFrame) and 'snp_beta' in beta_df.columns else False
    index_contains_snps =  True if isinstance(beta_df, pd.DataFrame) and beta_df.index.astype(str).startswith('rs').sum() > 0 else False
    return (contains_snps_column, index_contains_snps)

def infer_strain(beta_df_or_filepath, manifest=None):
    """Uses SNPS to identify the mouse strain (among 36 possibilities), using an internal lookup ref table from sesame.

argument:
    beta_df_or_filepath
        note that beta_df does not contain the snps, but you can provide access to a control_probes.pkl file
        and it will load and pull the snps for analysis. Or, if you use `methylcheck.load(<path>, format='beta_csv')`
        the beta_df WILL contain snps.
    manifest (default: None)
        It load the mouse manifest by default. But if you want to provide a custom manifest, you can.

.. note::
   This function calculates *Variant allele frequency (VAF)* in an intermediate step:
   - VAF is the percentage of sequence reads observed matching a specific DNA variant divided by the overall coverage at that locus.
   - VAF is a surrogate measure of the proportion of DNA molecules in the original specimen carrying the variant.



Possible Matching Strains:

    'DBA_1J', 'DBA_2J', 'AKR_J', 'C57L_J', '129S5SvEvBrd', 'BALB_cJ',
    'C57BL_10J', 'BTBR_T+_Itpr3tf_J', 'LP_J', 'A_J', '129S1_SvImJ', 'RF_J',
    'LEWES_EiJ', 'PWK_PhJ', 'BUB_BnJ', 'SPRET_EiJ', 'MOLF_EiJ', 'NZB_B1NJ',
    'NZO_HlLtJ', 'NZW_LacJ', 'KK_HiJ', '129P2_OlaHsd', 'C3H_HeJ', 'WSB_EiJ',
    'CBA_J', 'C3H_HeH', 'NOD_ShiLtJ', 'C57BR_cdJ', 'CAST_EiJ',
    'ZALENDE_EiJ', 'C58_J', 'C57BL_6NJ', 'ST_bJ', 'I_LnJ', 'SEA_GnJ', 'FVB_NJ'
    """
    if isinstance(beta_df_or_filepath, (Path, str)):
        fp = Path(beta_df_or_filepath)
        # Complicated because diff versions of methylprep put SNPs in different places.
        # Can either be part of control_probes, or a df loaded from CSVs using methylcheck.load
        if fp.exists() and fp.is_file():
            if '.pkl' in fp.suffixes and 'control_probes' in fp.name:
                beta_df = control_df_snps(fp)
            elif '.pkl' in fp.suffixes:
                beta_df = pd.read_pickle(fp)
                (contains_snps_column, index_contains_snps) = _dataframe_contains_snps(beta_df)
                if contains_snps_column is False or index_contains_snps is False:
                    LOGGER.warning(f"Detected a pickle that is not control_probes; dataframe structure unclear; contains a snps column: {contains_snps_column}; index contains snps: {index_contains_snps}")
            elif '.csv' in fp.suffxies:
                beta_df = pd.read_csv(fp)
                (contains_snps_column, index_contains_snps) = _dataframe_contains_snps(beta_df)
                if contains_snps_column is False or index_contains_snps is False:
                    LOGGER.warning(f"Detected a CSV; dataframe structure unclear; contains a snps column: {contains_snps_column}; index contains snps: {index_contains_snps}")
            else:
                raise ValueError(f"Your filepath ({beta_df_or_filepath}) must be either a csv or pkl file.")
        else:
            raise ValueError(f"Your filepath ({beta_df_or_filepath}) is not a valid file.")
    elif isinstance(beta_df_or_filepath, pd.DataFrame):
        beta_df = beta_df_or_filepath
        (contains_snps_column, index_contains_snps) = _dataframe_contains_snps(beta_df)
        if contains_snps_column is False or index_contains_snps is False:
            LOGGER.warning(f"Detected a dataframe; structure unclear; contains a snps column: {contains_snps_column}; index contains snps: {index_contains_snps}")
    elif isinstance(beta_df_or_filepath, pd.Series): # one sample case
        beta_df = pd.DataFrame(beta_df_or_filepath)
    elif hasattr(beta_df_or_filepath, "__len__"):
        beta_df = pd.DataFrame(beta_df_or_filepath) # not sure how probe names are included with this version
        LOGGER.warning(f"Detected as listlike object; provide a Series or DataFrame instead with probe names (IlmnID) in the index.")
    else:
        raise ValueError(f"Dataframe or filepath not recognized")

    vafs = mouse_beta_to_AF(beta_df, manifest=manifest)

    try:
        with resources.path(pkg_namespace, 'mouse_strain_snps.csv') as table_filepath:
            strain_snp_table = pd.read_csv(table_filepath).set_index('Unnamed: 0')
    except: # python < 3.7
        table_filepath = pkg_resources.resource_filename(pkg_namespace, 'mouse_strain_snps.csv')
        strain_snp_table = pd.read_csv(table_filepath).set_index('Unnamed: 0')
    strain_snp_table.index.name = 'IlmnID' # columns are 36 mouse strain names
    # 1 -- get only matching probes # probes <- intersect(names(vafs), rownames(strain_snp_table))
    # print(f"mouse strain snp data: {strain_snp_table.shape}")
    ## -- these commented lines decrease the output pval significantly (0.09 vs 0.40)
    # id_probes = vafs.index & strain_snp_table.index
    # strain_snp_table = strain_snp_table.loc[id_probes]
    # vafs = vafs.loc[ id_probes ]
    # print(f"{strain_snp_table.shape} & {vafs.shape}")
    # 2 -- set NaNs to 0.5
    vafs[ vafs.isna() ] = 0.5 # is this still valid as a method in latest pandas?
    # 3 -- dnorm() equiv is prob-density-funct norm.pdf()

    if isinstance(beta_df, pd.DataFrame):
        results = {}
        for sample in vafs: # probe is a series with all sample beta values for a given probe.
            bb = pd.DataFrame(norm.pdf(pd.DataFrame(vafs[sample]), loc=strain_snp_table.mean(), scale=0.8))
            bb.index = vafs.index
            bb = bb.rename(columns={k:v for k,v in enumerate(strain_snp_table.columns)})
            bb_log_likelihood = np.log(bb).sum() # these are negative numbers
            probs = np.exp(bb_log_likelihood - max(bb_log_likelihood)) # these are normalized probabilities with 1.00 being the highest
            best = np.argmax(probs) # best.index <- which.max(probs) <-- get position of highest probability ...
            strain_name = strain_snp_table.columns[best] # ... corresponding to Nth column in strain table
            results[sample] = {
                "best": strain_name,
                "pval": (sum(probs) - probs[best]) / sum(probs),
                "probs": probs/sum(probs)
                }
        return results

    else:
        bb = pd.DataFrame(norm.pdf(vafs, loc=strain_snp_table.mean(), scale=0.8))

        #bb <- vapply(probes, function(p) { #<--- vector apply this dnorm function, loops over each rs probe and all 36 strains as series
        #        dnorm(vafs[p], mean=strain_snp_table[p,], sd=0.8)
        #    }, numeric(ncol(strain_snp_table))) #<--- over length(N) probes

        # bb is a list of arrays; convert axes to probes(rows) X strains(cols)
        bb.index = vafs.index
        bb = bb.rename(columns={k:v for k,v in enumerate(strain_snp_table.columns)})

        #bbloglik <- apply(bb, 1, function(x) sum(log(x),na.rm=TRUE)) <--- apply [sum(log(x)) removing NA] over bb, (in 1st column of matrix?)
        bb_log_likelihood = np.log(bb).sum() # these are negative numbers
        probs = np.exp(bb_log_likelihood - max(bb_log_likelihood)) # these are normalized probabilities with 1.00 being the highest
        best = np.argmax(probs) # best.index <- which.max(probs) <-- get position of highest probability ...
        strain_name = strain_snp_table.columns[best] # ... corresponding to Nth column in strain table
        return {
            "best": strain_name,
            "pval": (sum(probs) - probs[best]) / sum(probs),
            "probs": probs/sum(probs)
            }


""" RESEARCH ON THIS
Infer strain information for mouse array. This will return a list containing
the best guess, p-value of the best guess, and probabilities of all strains.
Internally, the function converts the beta values to variant allele
frequencies (VAFs). It should be noted that since variant allele frequency is not
always measured as the M-allele, one needs to flip the beta values for some
probes to calculate variant allele frequency. (hence the "toFlip" step below. It reads the "design" column in
manifest)

# to load one mouse sample for testing strain in R
mouse <- read.csv(file= '/Volumes/LEGX/55085/55085_MURMETVEP/204375590039/204375590039_R01C01_processed.csv')
ms <- data.frame(mouse['beta_value'], mouse['IlmnID'], row.names='IlmnID')
# within sesame:
inferStrain(ms)

# need to copy the index into the dataframe first. map/apply won't pass the index value in otherwise
df = df[ df.IlmnID.str.startswith('rs') ]
df[ (df.design.str.contains('rs')) ] #<--- works, but don't compare to index

# HARDER but precise
df['IlmnID'] = df.index
df.design.map( lambda x: True if df.index.name in x else False )
where df is the mouse manifest.data_frame

#1 - get 'rs' probes from mouse manifest
# flip AF based on manifest annotation
# -- this pulls the 'design' column off manifest for 'rs' probes (a bunch of strings in pd.Series)
# design = GenomicRanges::mcols(mft)[['design']]
# -- this drops 'rs' probes if the 'design' column does not contain the 'rsXXXXXX' probe name in it.
# toFlip = !setNames(as.logical(substr(
#        design, nchar(design), nchar(design))), names(mft))
# -- filter the input betas to only 'rs' probes (that match the rsXXXX name for that probe) -- these must be flipped
# vafs = betas[grep('^rs', names(betas))]
# -- assign to vafs a values (1 - x)???
# vafs[toFlip[names(vafs)]] = 1-vafs[toFlip[names(vafs)]]
# returns the beta values, with some flipped, based on manifest, for proper strain ID.

MISSING MOUSE SNPS

213 found in control_probes.pkl:

['rs108256820_TC21', 'rs108270880_TC21', 'rs108708364_TC21', 'rs213041055_TC21', 'rs213308142_TC21', 'rs213638876_BC21', 'rs214721427_BC21', 'rs215033978_BC21', 'rs215246860_TC21', 'rs215310912_TC21', 'rs217747943_BC21', 'rs217896222_BC21', 'rs218141243_TC21', 'rs218321426_BC21', 'rs219242535_TC21', 'rs219356631_TC21', 'rs219753110_BC21', 'rs219988166_BC21', 'rs220569822_TC21', 'rs220599579_TC21', 'rs220976950_BC21', 'rs221474789_TC21', 'rs221586919_BC21', 'rs222875802_TC21', 'rs223418751_TC21', 'rs224166544_TC21', 'rs224474798_TC21', 'rs225111647_TC21', 'rs226075615_BC21', 'rs227235471_BC21', 'rs227257909_BC21', 'rs227343925_TC21', 'rs228378915_BC21', 'rs228395385_BC21', 'rs228546982_TC21', 'rs228633605_TC21', 'rs229241225_BC21', 'rs229575846_BC21', 'rs229907807_BC21', 'rs229913504_BC21', 'rs230272606_TC21', 'rs230528105_TC21', 'rs230802589_BC21', 'rs230812213_BC21', 'rs230933543_BC21', 'rs231569683_BC21', 'rs231783152_BC21', 'rs231864851_BC21', 'rs232251489_TC21', 'rs232456961_BC21', 'rs233959756_TC21', 'rs234134949_BC21', 'rs234657554_TC21', 'rs234825256_TC21', 'rs235357786_TC21', 'rs236594123_TC21', 'rs236624633_BC21', 'rs236731053_BC21', 'rs237581131_TC21', 'rs238381919_BC21', 'rs239634423_BC21', 'rs240101497_BC21', 'rs240421468_TC21', 'rs241079868_BC21', 'rs241169121_BC21', 'rs241373111_TC21', 'rs241697348_BC21', 'rs242382401_BC21', 'rs242930132_BC21', 'rs243106940_BC21', 'rs243427388_TC21', 'rs243789120_TC21', 'rs243873738_TC21', 'rs243890462_BC21', 'rs244808377_TC21', 'rs244847823_TC21', 'rs245307400_TC21', 'rs245545222_TC21', 'rs246227460_BC21', 'rs246625745_TC21', 'rs246917109_BC21', 'rs247394039_BC21', 'rs247490468_TC21', 'rs247569630_TC21', 'rs247636862_TC21', 'rs247824387_BC21', 'rs247974563_TC21', 'rs248191939_BC21', 'rs248892702_TC21', 'rs249350499_BC21', 'rs250218667_BC21', 'rs251745524_TC21', 'rs252470098_TC21', 'rs252749385_BC21', 'rs252823583_BC21', 'rs252985227_BC21', 'rs253330544_TC21', 'rs253397710_BC21', 'rs254463362_TC21', 'rs254673009_BC21', 'rs254769212_BC21', 'rs254962283_TC21', 'rs255474359_BC21', 'rs255474359_BC22', 'rs256556599_TC21', 'rs256836054_TC21', 'rs256858903_TC21', 'rs256923082_BC21', 'rs257379098_TC21', 'rs257429108_TC21', 'rs257595325_BC21', 'rs258027624_BC21', 'rs258676951_TC21', 'rs258892227_TC21', 'rs259183823_TC21', 'rs259517289_BC21', 'rs260931187_TC21', 'rs261472181_TC21', 'rs261531602_TC21', 'rs262998696_TC21', 'rs263669189_BC21', 'rs265113641_BC21', 'rs265515652_BC21', 'rs265785977_TC21', 'rs27066649_TC21', 'rs27665449_BC21', 'rs27932276_TC21', 'rs28031290_TC21', 'rs28104183_TC21', 'rs28159258_BC21', 'rs28161401_BC21', 'rs29072964_TC21', 'rs29095631_TC21', 'rs29097237_TC21', 'rs29156031_BC21', 'rs29160242_TC21', 'rs29175830_TC21', 'rs29242322_BC21', 'rs29278211_BC21', 'rs29493699_TC21', 'rs29608214_TC21', 'rs29749171_BC21', 'rs29779002_TC21', 'rs29816695_TC21', 'rs29950245_TC21', 'rs30480269_TC21', 'rs30667335_BC21', 'rs30692090_TC21', 'rs30746315_TC21', 'rs30817317_BC21', 'rs30984204_BC21', 'rs31064702_TC21', 'rs31179078_BC21', 'rs31369752_TC21', 'rs31413653_TC21', 'rs31563231_BC21', 'rs31602860_TC21', 'rs31626506_BC21', 'rs31786682_BC21', 'rs31792310_TC21', 'rs32062798_TC21', 'rs32086004_TC21', 'rs32157573_BC21', 'rs32430537_BC21', 'rs33036540_BC21', 'rs33233558_TC21', 'rs33381927_BC21', 'rs33388595_BC21', 'rs33715364_BC21', 'rs36342546_BC21', 'rs36397114_TC21', 'rs3654986_TC21', 'rs36617736_BC21', 'rs3668421_TC21', 'rs37087486_BC21', 'rs3718745_BC21', 'rs37868712_BC21', 'rs38137123_TC21', 'rs38597635_TC21', 'rs387510518_BC21', 'rs39674167_BC21', 'rs40125798_BC21', 'rs4211703_BC21', 'rs46270449_TC21', 'rs46462788_TC21', 'rs46517650_BC21', 'rs46715553_TC21', 'rs46751997_BC21', 'rs47058788_BC21', 'rs47203680_TC21', 'rs47515890_TC21', 'rs47896085_TC21', 'rs48211255_BC21', 'rs48535333_BC21', 'rs48868852_BC21', 'rs49174199_TC21', 'rs49221881_BC21', 'rs49486182_TC21', 'rs49658760_BC21', 'rs49775858_BC21', 'rs50049494_TC21', 'rs50242871_BC21', 'rs50571417_TC21', 'rs50615896_TC21', 'rs51172188_BC21', 'rs51346161_TC21', 'rs51795756_TC21', 'rs52054624_BC21', 'rs578266421_BC21', 'rs583116188_BC21', 'rs584222858_BC21', 'rs6153119_BC21', 'rs6167777_TC21']

-- out Counter({'TC21': 107, 'BC21': 105, 'BC22': 1})
vs
-- man Counter({'BC11': 346, 'TC11': 346, 'TC21': 285, 'BC21': 284, 'TC12': 100, 'BC12': 81, 'BO11': 21, 'TO11': 18, 'BC22': 5})
(could it be issue with duplicate readings per probe?)

1486 in manifest:

['rs1-101008622_BC11', 'rs1-146717652_BC11', 'rs1-146717652_TC11', 'rs11-118757022_TC11', 'rs11-4183579_BC11', 'rs11-4183579_TC11', 'rs11-72715146_TC11', 'rs11-81409699_BC11', 'rs11-9451399_TC11', 'rs11-9451399_TC12', 'rs12-45422860_BC11', 'rs13-56442774_TC11', 'rs13-56442774_TC12', 'rs14-102540269_BC11', 'rs14-102540269_BC12', 'rs14-102540269_TC11', 'rs14-105754376_BC11', 'rs14-105754376_BC12', 'rs14-29220639_TC11', 'rs14-57046559_BC11', 'rs14-57046559_BC12', 'rs14-68848815_TC11', 'rs14-68848815_TC12', 'rs16-79651326_BC11', 'rs16-79651326_BC12', 'rs16-93198125_TC11', 'rs17-49346226_BC11', 'rs17-49346226_BC12', 'rs17-49346226_TC11', 'rs17-65441146_TC11', 'rs17-65441146_TC12', 'rs17-90509437_TC11', 'rs17-90509437_TC12', 'rs2-65531693_BC11', 'rs2-79399607_TC11', 'rs2-79399607_TC12', 'rs3-104718631_BC11', 'rs3-73115385_TC11', 'rs3-73115385_TC12', 'rs4-27201098_BC11', 'rs4-29382142_BC11', 'rs4-29382142_BC12', 'rs4-31167599_BC11', 'rs4-31167599_TC11', 'rs4-76042086_TC11', 'rs4-76042086_TC12', 'rs4-83461976_BC11', 'rs4-83461976_TC11', 'rs4-90920216_BC11', 'rs4-90920216_BC12', 'rs4-9998685_BC11', 'rs4-9998685_TC11', 'rs5-101487991_TC11', 'rs5-101487991_TC12', 'rs5-20101526_BC11', 'rs5-52164994_BC11', 'rs5-52164994_TC11', 'rs5-52164994_TC12', 'rs6-11530051_BC11', 'rs6-11530051_BO11', 'rs7-123288709_TC11', 'rs7-14467849_BC11', 'rs8-128482818_TC11', 'rs8-128482818_TC12', 'rs8-3871958_BC11', 'rs8-3871958_TC11', 'rs8-46243839_BC11', 'rs8-46243839_BC12', 'rs8-56855732_TC11', 'rs8-86331371_BC11', 'rs8-86331371_BC12', 'rs9-122111793_BC11', 'rs9-122111793_BC12', 'rs9-123336588_BC11', 'rs9-24938681_BC11', 'rs9-24938681_BC12', 'rs9-41473787_TC11', 'rs9-49486682_BC11', 'rs9-53259736_BC11', 'rs9-86749507_BC11', 'rs9-86749507_TC11', 'rs9-86749507_TC12', 'rsX-139395999_BC11', 'rs108256820_TC11', 'rs108256820_TC12', 'rs108270880_BC11', 'rs108270880_TC11', 'rs108708364_TC11', 'rs108708364_TC12', 'rs108787510_BC11', 'rs108867985_BC11', 'rs13484029_BC11', 'rs13484029_BC12', 'rs13484029_TC11', 'rs211820359_BC11', 'rs211820359_BC12', 'rs212651458_BC11', 'rs213009914_BC11', 'rs213261634_TC11', 'rs213261634_TC12', 'rs213308142_TC11', 'rs213638876_BC11', 'rs213638876_BC12', 'rs213725211_TC11', 'rs213832426_TC11', 'rs213832426_TC12', 'rs214072736_BC11', 'rs214072736_TC11', 'rs214537044_BC11', 'rs214603785_TC11', 'rs214603785_TC12', 'rs214645656_TC11', 'rs214661156_BC11', 'rs214721427_BC11', 'rs214839557_TC11', 'rs214866691_TC11', 'rs214866691_TC12', 'rs214969250_TC11', 'rs214969250_TC12', 'rs215033978_TC11', 'rs215246860_TC11', 'rs215246860_TC12', 'rs215310912_BC11', 'rs215310912_TC11', 'rs215425012_BC11', 'rs215425012_BC12', 'rs216352409_BC11', 'rs216352409_TC11', 'rs216352409_TC12', 'rs217747943_BC11', 'rs217747943_BC12', 'rs217896222_BC11', 'rs217942576_TC11', 'rs218141243_TC11', 'rs218321426_BC11', 'rs218606137_BC11', 'rs218606137_TC11', 'rs219153217_TC11', 'rs219242535_BC11', 'rs219242535_TC11', 'rs219356631_TC11', 'rs219356631_TC12', 'rs219585415_TC11', 'rs219753110_BC11', 'rs219958685_BC11', 'rs219988166_BC11', 'rs219988166_BC12', 'rs220040456_BC11', 'rs220086268_TC11', 'rs220569822_TC11', 'rs220599579_BC11', 'rs220599579_TC11', 'rs220599579_TC12', 'rs220976950_BC11', 'rs220976950_TC11', 'rs221388542_BC11', 'rs221388542_BC12', 'rs221440008_TC11', 'rs221474789_BC11', 'rs221474789_TC11', 'rs221474789_TC12', 'rs221570958_BC11', 'rs221579898_BC11', 'rs221579898_TC11', 'rs221586919_BC11', 'rs221586919_BO11', 'rs221755933_BC11', 'rs222653690_BC11', 'rs222735508_BC11', 'rs222735508_BC12', 'rs222875802_TC11', 'rs222920567_BC11', 'rs222920567_TC11', 'rs223092379_BC11', 'rs223092379_BC12', 'rs223418751_TC11', 'rs223418751_TC12', 'rs223425758_BC11', 'rs223682945_TC11', 'rs223682945_TO11', 'rs224166544_TC11', 'rs224474798_TC11', 'rs224474798_TC12', 'rs225111647_TC11', 'rs225856342_BC11', 'rs225856342_BC12', 'rs226075615_BC11', 'rs226108635_BC11', 'rs226108635_TC11', 'rs226108635_TC12', 'rs226357209_BC11', 'rs226357209_TC11', 'rs226813272_BC11', 'rs226813272_TC11', 'rs227235471_BC11', 'rs227235471_BO11', 'rs227257909_BC11', 'rs227343925_TC11', 'rs227343925_TC12', 'rs227456166_BC11', 'rs227456166_TC11', 'rs228014876_BC11', 'rs228040302_TC11', 'rs228040302_TC12', 'rs228378915_BC11', 'rs228378915_TC11', 'rs228395385_BC11', 'rs228395385_BC12', 'rs228395385_TC11', 'rs228472106_BC11', 'rs228472106_TC11', 'rs228472106_TC12', 'rs228546982_TC11', 'rs228633605_BC11', 'rs228633605_TC11', 'rs228633605_TC12', 'rs228827083_BC11', 'rs228827083_TC11', 'rs228945691_BC11', 'rs229015623_BC11', 'rs229210769_TC11', 'rs229241225_BC11', 'rs229473485_TC11', 'rs229575846_BC11', 'rs229575846_BC12', 'rs229907807_BC11', 'rs229913504_BC11', 'rs229913504_BC12', 'rs230265134_BC11', 'rs230272606_BC11', 'rs230528105_BC11', 'rs230528105_TC11', 'rs230713354_BC11', 'rs230802589_BO11', 'rs230812213_BC11', 'rs230812213_BO11', 'rs230933543_BC11', 'rs230933543_BO11', 'rs231783152_BC11', 'rs231864851_BC11', 'rs232251489_TC11', 'rs232456961_BC11', 'rs232456961_BC12', 'rs233104467_BC11', 'rs233104467_BO11', 'rs233104467_TC11', 'rs233195649_TC11', 'rs233896555_BC11', 'rs233896555_TC11', 'rs233959756_TC11', 'rs234134949_BC11', 'rs234134949_BC12', 'rs234243290_TC11', 'rs234657554_TC11', 'rs234689890_TC11', 'rs234825256_TC11', 'rs234834208_BC11', 'rs234936391_TC11', 'rs234936391_TC12', 'rs235353808_BC11', 'rs235353808_TC11', 'rs235357786_TC11', 'rs235486993_BC11', 'rs235802917_BC11', 'rs236081924_BC11', 'rs236081924_BC12', 'rs236594123_BC11', 'rs236594123_TC11', 'rs236594123_TC12', 'rs236624633_BC11', 'rs236632242_BC11', 'rs236632242_BC12', 'rs236680206_BC11', 'rs236680206_BC12', 'rs236731053_BC11', 'rs236989564_BC11', 'rs237581131_BC11', 'rs237581131_TC11', 'rs238381919_BC11', 'rs238682393_BC11', 'rs238682393_TC11', 'rs238722708_BC11', 'rs238722708_TC11', 'rs238891333_BC11', 'rs238899583_TC11', 'rs238899583_TO11', 'rs239118269_BC11', 'rs239118269_BC12', 'rs239634423_BC11', 'rs240049454_TC11', 'rs240049454_TC12', 'rs240101497_BC11', 'rs240101497_TC11', 'rs240174762_BC11', 'rs240174762_BO11', 'rs240421468_TC11', 'rs241079868_BC11', 'rs241079868_TC11', 'rs241169121_BC11', 'rs241373111_TC11', 'rs241466001_BC11', 'rs241466001_TC11', 'rs241697348_BC11', 'rs242382401_BC11', 'rs242382401_BO11', 'rs242399136_BC11', 'rs242399136_TC11', 'rs242930132_BC11', 'rs242930132_TC11', 'rs242957558_BC11', 'rs243106940_BC11', 'rs243168225_BC11', 'rs243427388_TC11', 'rs243428925_BC11', 'rs243428925_BC12', 'rs243789120_TC11', 'rs243789120_TC12', 'rs243873738_BC11', 'rs243873738_TC11', 'rs243873738_TC12', 'rs243976225_BC11', 'rs243976225_BC12', 'rs243976225_TC11', 'rs244056657_BC11', 'rs244300579_TC11', 'rs244808377_TC11', 'rs244808377_TC12', 'rs244847823_BC11', 'rs244847823_TC11', 'rs244908367_BC11', 'rs244963628_BC11', 'rs244963628_BC12', 'rs244963628_TC11', 'rs245307400_TC11', 'rs245307400_TC12', 'rs245545222_TC11', 'rs245545222_TC12', 'rs245805881_TC11', 'rs245995211_BC11', 'rs245995211_TC11', 'rs245995211_TC12', 'rs246127201_TC11', 'rs246127201_TC12', 'rs246227460_BC11', 'rs246262673_TC11', 'rs246625745_BC11', 'rs246625745_TC11', 'rs246839600_TC11', 'rs246917109_BO11', 'rs247282477_BC11', 'rs247394039_BC11', 'rs247447149_TC11', 'rs247447149_TC12', 'rs247542741_BC11', 'rs247542741_TC11', 'rs247569630_TC11', 'rs247636862_TC11', 'rs247664967_BC11', 'rs247824387_BC11', 'rs247974563_TC11', 'rs248191939_BC11', 'rs248191939_TC11', 'rs248221995_BC11', 'rs248221995_TC11', 'rs248476972_TC11', 'rs248699051_BC11', 'rs248892702_BC11', 'rs249350499_BC11', 'rs249350499_BC12', 'rs249746634_BC11', 'rs250170981_TC11', 'rs250170981_TC12', 'rs250218667_BC11', 'rs250218667_BC12', 'rs250218667_TC11', 'rs250466379_TC11', 'rs250595130_TC11', 'rs250971740_TC11', 'rs251378371_TC11', 'rs251745524_BC11', 'rs251745524_TC11', 'rs251745524_TO11', 'rs251915108_TC11', 'rs251915108_TC12', 'rs251928543_BC11', 'rs252101019_BC11', 'rs252470098_TC11', 'rs252470098_TC12', 'rs252606438_TC11', 'rs252749385_BC11', 'rs252815897_TC11', 'rs252815897_TC12', 'rs252915005_TC11', 'rs252985227_BC11', 'rs252985227_TC11', 'rs253167024_BC11', 'rs253167024_BO11', 'rs253178866_BC11', 'rs253178866_BC12', 'rs253330544_BC11', 'rs253330544_TC11', 'rs253330544_TC12', 'rs253397710_BC11', 'rs253397710_TC11', 'rs253697629_BC11', 'rs253697629_BC12', 'rs253699085_BC11', 'rs253699085_TC11', 'rs254050478_TC11', 'rs254050478_TC12', 'rs254463362_TC11', 'rs254558324_BC11', 'rs254558324_BC12', 'rs254673009_BC11', 'rs254673009_TC11', 'rs254769212_BC11', 'rs254962283_TC11', 'rs254962283_TC12', 'rs255030006_BC11', 'rs255030006_BO11', 'rs255030006_TC11', 'rs255202501_BC11', 'rs255202501_BC12', 'rs255216138_TC11', 'rs255216138_TC12', 'rs255314446_BC11', 'rs255314446_TC11', 'rs255474359_BC11', 'rs255647337_TO11', 'rs255694468_BC11', 'rs255694468_TC11', 'rs255702174_TC11', 'rs255702174_TC12', 'rs255815160_BC11', 'rs255815160_BC12', 'rs256284328_BC11', 'rs256556599_TC11', 'rs256709818_BC11', 'rs256709818_BC12', 'rs256709818_TC11', 'rs256716102_TC11', 'rs256801086_BC11', 'rs256801086_BC12', 'rs256836054_TC11', 'rs256858903_TC11', 'rs256858903_TC12', 'rs256923082_BC11', 'rs257179659_TC11', 'rs257206219_TC11', 'rs257206219_TC12', 'rs257351463_TC11', 'rs257351463_TC12', 'rs257379098_BC11', 'rs257379098_TC11', 'rs257379098_TC12', 'rs257429108_TC11', 'rs257429108_TC12', 'rs257587236_BC11', 'rs257595325_BC11', 'rs258027624_BC11', 'rs258027624_BC12', 'rs258302806_TC11', 'rs258337496_TC11', 'rs258337496_TC12', 'rs258676951_TC11', 'rs258892227_TC11', 'rs258902364_TC11', 'rs259149286_BC11', 'rs259149286_BC12', 'rs259149286_TC11', 'rs259159595_BC11', 'rs259178054_BC11', 'rs259178054_TC11', 'rs259183823_TC11', 'rs259233921_BC11', 'rs259233921_TC11', 'rs259254565_BC11', 'rs259254565_TC11', 'rs259355761_BC11', 'rs259355761_TC11', 'rs259358660_TC11', 'rs259517289_BC11', 'rs259517289_BC12', 'rs259540824_BC11', 'rs259540824_BC12', 'rs259540824_TC11', 'rs259964007_BC11', 'rs259964007_TC11', 'rs260931187_TC11', 'rs260931187_TC12', 'rs261034204_BC11', 'rs261034204_TC11', 'rs261100115_BC11', 'rs261100115_BC12', 'rs261434524_BC11', 'rs261434524_BO11', 'rs261472181_TC11', 'rs261531602_TC11', 'rs261766351_BC11', 'rs261766351_TC11', 'rs261766351_TC12', 'rs263472428_TC11', 'rs263489370_TC11', 'rs263489370_TO11', 'rs263566580_BC11', 'rs263566580_BC12', 'rs263669189_BC11', 'rs264029118_BC11', 'rs264029118_BC12', 'rs264131445_TC11', 'rs264131445_TC12', 'rs264144035_BC11', 'rs264144035_BO11', 'rs264283936_BC11', 'rs264283936_TC11', 'rs264337053_BC11', 'rs264840505_BC11', 'rs264841685_BC11', 'rs264841685_BC12', 'rs265036461_BC11', 'rs265036461_TC11', 'rs265036461_TC12', 'rs265113641_BC11', 'rs265189229_TC11', 'rs265189229_TC12', 'rs265202640_BC11', 'rs265202640_TC11', 'rs265515652_BC11', 'rs265515652_BC12', 'rs265785977_BC11', 'rs265785977_TC11', 'rs265785977_TC12', 'rs265921441_BC11', 'rs265921441_TC11', 'rs266047304_TC11', 'rs26847171_TC11', 'rs26864015_BC11', 'rs26897876_TC11', 'rs26915261_BC11', 'rs26915261_BC12', 'rs27065840_BC11', 'rs27066470_BC11', 'rs27066470_BC12', 'rs27066649_TC11', 'rs27201936_TC11', 'rs27218157_TC11', 'rs27266822_BC11', 'rs27266822_TC11', 'rs27307306_TC11', 'rs27380067_TC11', 'rs27380067_TO11', 'rs27410792_TC11', 'rs27439726_TC11', 'rs27439726_TC12', 'rs27458378_BC11', 'rs27469075_TC11', 'rs27469075_TC12', 'rs27469737_TC11', 'rs27569332_TC11', 'rs27647916_BC11', 'rs27665449_BC11', 'rs27670649_BC11', 'rs27670649_TC11', 'rs27703534_TC11', 'rs27703534_TO11', 'rs27904525_TC11', 'rs27930372_BC11', 'rs27932276_BC11', 'rs27932276_TC11', 'rs27932276_TO11', 'rs28031290_BC11', 'rs28031290_TC11', 'rs28159258_BC11', 'rs28161401_BC11', 'rs28173501_TC11', 'rs28173501_TC12', 'rs28179788_BC11', 'rs28327594_TC11', 'rs28331298_BC11', 'rs29041768_TC11', 'rs29056176_BC11', 'rs29056176_TC11', 'rs29056176_TC12', 'rs29072964_BC11', 'rs29072964_TC11', 'rs29095631_TC11', 'rs29097237_TC11', 'rs29105385_BC11', 'rs29105385_TC11', 'rs29107060_TC11', 'rs29152857_BC11', 'rs29152857_BC12', 'rs29152857_TC11', 'rs29156031_BC11', 'rs29175830_BC11', 'rs29175830_TC11', 'rs29178198_TC11', 'rs29192233_BC11', 'rs29203451_TC11', 'rs29203451_TC12', 'rs29208028_BC11', 'rs29215937_TC11', 'rs29215937_TC12', 'rs29242322_BC11', 'rs29272296_TC11', 'rs29272296_TC12', 'rs29278211_BC11', 'rs29278211_BO11', 'rs29281557_BC11', 'rs29281557_TC11', 'rs29281557_TC12', 'rs29285409_BC11', 'rs29285409_BO11', 'rs29305617_BC11', 'rs29305617_TC11', 'rs29311435_BC11', 'rs29311435_BC12', 'rs29431399_BC11', 'rs29431399_TC11', 'rs29493699_BC11', 'rs29493699_TC11', 'rs29493699_TC12', 'rs29494376_TC11', 'rs29494376_TC12', 'rs29509115_TC11', 'rs29509115_TO11', 'rs29558397_TC11', 'rs29566866_TC11', 'rs29608214_TC11', 'rs29632888_BC11', 'rs29632888_TC11', 'rs29632888_TC12', 'rs29717428_BC11', 'rs29717428_BC12', 'rs29726863_TC11', 'rs29749171_BC11', 'rs29749171_TC11', 'rs29773309_TC11', 'rs29773309_TO11', 'rs29779002_TC11', 'rs29779002_TC12', 'rs29801485_BC11', 'rs29801485_BC12', 'rs29810546_BC11', 'rs29810546_BO11', 'rs29816695_TC11', 'rs29950245_TC11', 'rs29977096_TC11', 'rs30179612_BC11', 'rs30179612_TC11', 'rs30179612_TC12', 'rs30289272_BC11', 'rs30289272_TC11', 'rs30289272_TC12', 'rs30312012_BC11', 'rs30312012_BO11', 'rs30381374_BC11', 'rs30381374_BC12', 'rs30468722_TC11', 'rs30468722_TC12', 'rs30544720_BC11', 'rs30544720_BC12', 'rs30544720_TC11', 'rs30667335_BC11', 'rs30692090_TC11', 'rs30746315_TC11', 'rs30817317_BC11', 'rs30836183_TC11', 'rs30836183_TC12', 'rs30984204_BC11', 'rs30984204_BC12', 'rs31034349_TC11', 'rs31034349_TC12', 'rs31064702_TC11', 'rs31091714_TC11', 'rs31091714_TC12', 'rs31179078_BC11', 'rs31179078_BC12', 'rs31241000_BC11', 'rs31241000_TC11', 'rs31338587_TC11', 'rs31349762_BC11', 'rs31369752_BC11', 'rs31369752_TC11', 'rs31413653_TC11', 'rs31558350_BC11', 'rs31558350_BC12', 'rs31563231_BC11', 'rs31596046_BC11', 'rs31602860_BC11', 'rs31602860_TC11', 'rs31602860_TC12', 'rs31626506_BC11', 'rs31693830_TC11', 'rs31766592_BC11', 'rs31766592_BC12', 'rs31786682_BC11', 'rs31792310_TC11', 'rs31842080_BC11', 'rs31842080_TC11', 'rs31895645_TC11', 'rs32062798_BC11', 'rs32086004_TC11', 'rs32153474_TC11', 'rs32255258_TC11', 'rs32299283_BC11', 'rs32355522_BC11', 'rs32355522_TC11', 'rs32398298_BC11', 'rs32398298_TC11', 'rs32430537_BC11', 'rs32430537_BC12', 'rs32478480_BC11', 'rs32531309_BC11', 'rs32531309_BO11', 'rs32703576_BC11', 'rs32703576_BC12', 'rs32728065_BC11', 'rs32728065_TC11', 'rs32781835_TC11', 'rs32781835_TO11', 'rs33036540_BC11', 'rs33109580_BC11', 'rs33109580_TC11', 'rs33233558_TC11', 'rs33233558_TC12', 'rs33362956_BC11', 'rs33362956_BC12', 'rs33381927_BC11', 'rs33388595_BC11', 'rs33388595_BC12', 'rs33388595_TC11', 'rs33605391_TC11', 'rs33605391_TC12', 'rs33622938_BC11', 'rs33622938_TC11', 'rs33664267_BC11', 'rs33695576_TC11', 'rs33695576_TC12', 'rs33715364_BC11', 'rs33752746_BC11', 'rs33779096_BC11', 'rs36334732_BC11', 'rs36334732_BC12', 'rs36334732_TC11', 'rs36342546_BC11', 'rs36382248_TC11', 'rs36397114_TC11', 'rs36397114_TC12', 'rs36402268_TC11', 'rs3654986_TC11', 'rs3654986_TC12', 'rs36605103_BC11', 'rs36617736_BC11', 'rs3667332_TC11', 'rs3668421_TC11', 'rs37087486_BC11', 'rs37115174_BC11', 'rs37115174_TC11', 'rs37152337_TC11', 'rs3718745_BC11', 'rs37324321_TC11', 'rs37324321_TC12', 'rs37398972_TC11', 'rs37454539_TC11', 'rs37454539_TO11', 'rs37660116_BC11', 'rs37660116_BC12', 'rs37660116_TC11', 'rs37841887_TC11', 'rs37841887_TO11', 'rs37868712_BC11', 'rs37868712_TC11', 'rs38137123_BC11', 'rs38137123_TC11', 'rs38502664_TC11', 'rs38502664_TC12', 'rs38514242_BC11', 'rs38514242_TC11', 'rs38597635_BC11', 'rs38597635_TC11', 'rs386894787_TC11', 'rs387058893_TC11', 'rs387058893_TO11', 'rs387510518_BC11', 'rs387546874_BC11', 'rs387546874_TC11', 'rs39674167_BC11', 'rs40125798_BC11', 'rs4152042_TC11', 'rs4190153_TC11', 'rs4211703_BC11', 'rs4211703_TC11', 'rs4221598_TC11', 'rs46079949_TC11', 'rs46079949_TC12', 'rs46270449_TC11', 'rs46462788_TC11', 'rs46517650_BC11', 'rs46563785_BC11', 'rs46563785_TC11', 'rs46646908_BC11', 'rs46646908_BC12', 'rs46715553_TC11', 'rs46715553_TC12', 'rs46751997_BC11', 'rs46751997_TC11', 'rs46771972_BC11', 'rs46771972_BC12', 'rs46907146_BC11', 'rs46907146_BC12', 'rs46907146_TC11', 'rs47058788_BC11', 'rs47164625_BC11', 'rs47164625_BO11', 'rs47203680_TC11', 'rs47415320_TC11', 'rs47415320_TC12', 'rs47435801_BC11', 'rs47515890_BC11', 'rs47515890_TC11', 'rs47515890_TC12', 'rs47896085_BC11', 'rs47896085_TC11', 'rs47896085_TO11', 'rs47959292_BC11', 'rs47959292_TC11', 'rs48211255_BC11', 'rs48211255_BC12', 'rs48284079_BC11', 'rs48284079_BC12', 'rs48488840_TC11', 'rs48521828_BC11', 'rs48521828_BC12', 'rs48535333_BC11', 'rs48539460_BC11', 'rs48539460_TC11', 'rs48687294_TC11', 'rs48687294_TC12', 'rs48792241_BC11', 'rs48794940_BC11', 'rs48794940_TC11', 'rs48807362_BC11', 'rs48868852_BC11', 'rs48868852_BC12', 'rs49174199_TC11', 'rs49221881_BC11', 'rs49265791_TC11', 'rs49265791_TO11', 'rs49302583_TC11', 'rs49302583_TC12', 'rs49448999_BC11', 'rs49448999_BC12', 'rs49486182_TC11', 'rs49545944_BC11', 'rs49560492_BC11', 'rs49621304_BC11', 'rs49658760_BC11', 'rs49716043_BC11', 'rs49716043_BO11', 'rs49775858_BC11', 'rs49775858_TC11', 'rs49811955_BC11', 'rs50049494_TC11', 'rs50242871_BC11', 'rs50242871_BC12', 'rs50315637_BC11', 'rs50372932_TC11', 'rs50571417_TC11', 'rs50615896_TC11', 'rs50706378_TC11', 'rs50706378_TC12', 'rs50731739_BC11', 'rs50731739_BC12', 'rs50801342_TC11', 'rs50801342_TC12', 'rs51153073_BC11', 'rs51153073_BC12', 'rs51153073_TC11', 'rs51172188_BC11', 'rs51334756_BC11', 'rs51334756_TC11', 'rs51346161_TC11', 'rs51346161_TC12', 'rs51465303_TC11', 'rs51534139_TC11', 'rs51543475_TC11', 'rs51543475_TC12', 'rs51623688_TC11', 'rs51623688_TO11', 'rs51702443_TC11', 'rs51717066_BC11', 'rs51717066_TC11', 'rs51795756_BC11', 'rs51795756_TC11', 'rs51841049_BC11', 'rs51841049_TC11', 'rs51867246_TC11', 'rs52054624_BC11', 'rs52570224_TC11', 'rs578535906_TC11', 'rs578535906_TO11', 'rs581280974_TC11', 'rs583116188_BC11', 'rs584222858_BC11', 'rs585463869_BC11', 'rs585463869_TC11', 'rs6153119_BC11', 'rs6153119_BC12', 'rs6167777_TC11', 'rs6167777_TC12', 'rs6183279_BC11', 'rs1-101008622_BC21', 'rs1-146717652_BC21', 'rs11-118757022_TC21', 'rs11-4183579_TC21', 'rs11-72715146_TC21', 'rs11-81409699_BC21', 'rs11-81409699_BC22', 'rs11-9451399_TC21', 'rs12-45422860_BC21', 'rs13-56442774_TC21', 'rs14-102540269_BC21', 'rs14-105754376_BC21', 'rs14-29220639_TC21', 'rs14-57046559_BC21', 'rs14-68848815_TC21', 'rs16-93198125_TC21', 'rs17-49346226_BC21', 'rs17-65441146_TC21', 'rs17-90509437_TC21', 'rs19-44422173_BC21', 'rs2-65531693_BC21', 'rs2-79399607_TC21', 'rs3-104718631_BC21', 'rs3-73115385_TC21', 'rs4-27201098_BC21', 'rs4-29382142_BC21', 'rs4-31167599_TC21', 'rs4-83461976_BC21', 'rs4-90920216_BC21', 'rs4-9998685_TC21', 'rs5-101487991_TC21', 'rs5-20101526_BC21', 'rs5-52164994_TC21', 'rs6-11530051_BC21', 'rs6-48454003_TC21', 'rs7-123288709_TC21', 'rs7-14467849_BC21', 'rs8-128482818_TC21', 'rs8-3871958_BC21', 'rs8-46243839_BC21', 'rs8-56855732_TC21', 'rs8-86331371_BC21', 'rs9-122111793_BC21', 'rs9-123336588_BC21', 'rs9-24938681_BC21', 'rs9-41473787_TC21', 'rs9-49486682_BC21', 'rs9-53259736_BC21', 'rs9-86749507_TC21', 'rsX-139395999_BC21', 'rs108256820_TC21', 'rs108270880_TC21', 'rs108708364_TC21', 'rs108787510_BC21', 'rs108867985_BC21', 'rs13484029_BC21', 'rs211820359_BC21', 'rs212651458_BC21', 'rs213009914_BC21', 'rs213041055_TC21', 'rs213261634_TC21', 'rs213308142_TC21', 'rs213638876_BC21', 'rs213725211_TC21', 'rs213832426_TC21', 'rs214072736_BC21', 'rs214537044_BC21', 'rs214603785_TC21', 'rs214645656_TC21', 'rs214661156_BC21', 'rs214721427_BC21', 'rs214839557_TC21', 'rs214866691_TC21', 'rs214969250_TC21', 'rs215033978_BC21', 'rs215246860_TC21', 'rs215310912_TC21', 'rs215425012_BC21', 'rs216352409_TC21', 'rs217747943_BC21', 'rs217896222_BC21', 'rs217942576_TC21', 'rs218141243_TC21', 'rs218321426_BC21', 'rs218606137_BC21', 'rs218830754_BC21', 'rs219153217_TC21', 'rs219242535_TC21', 'rs219356631_TC21', 'rs219753110_BC21', 'rs219958685_BC21', 'rs219988166_BC21', 'rs220040456_BC21', 'rs220086268_TC21', 'rs220569822_TC21', 'rs220599579_TC21', 'rs220976950_BC21', 'rs221388542_BC21', 'rs221440008_TC21', 'rs221474789_TC21', 'rs221570958_BC21', 'rs221579898_TC21', 'rs221586919_BC21', 'rs221755933_BC21', 'rs222653690_BC21', 'rs222735508_BC21', 'rs222875802_TC21', 'rs223092379_BC21', 'rs223418751_TC21', 'rs223425758_BC21', 'rs223682945_TC21', 'rs224166544_TC21', 'rs224474798_TC21', 'rs225111647_TC21', 'rs225856342_BC21', 'rs226075615_BC21', 'rs226108635_TC21', 'rs226357209_TC21', 'rs226813272_BC21', 'rs227235471_BC21', 'rs227257909_BC21', 'rs227343925_TC21', 'rs227456166_BC21', 'rs228014876_BC21', 'rs228040302_TC21', 'rs228378915_BC21', 'rs228395385_BC21', 'rs228472106_TC21', 'rs228546982_TC21', 'rs228633605_TC21', 'rs228945691_BC21', 'rs229015623_BC21', 'rs229241225_BC21', 'rs229473485_TC21', 'rs229575846_BC21', 'rs229907807_BC21', 'rs229913504_BC21', 'rs230265134_BC21', 'rs230272606_TC21', 'rs230528105_TC21', 'rs230647266_TC21', 'rs230713354_BC21', 'rs230802589_BC21', 'rs230812213_BC21', 'rs230933543_BC21', 'rs231569683_BC21', 'rs231569683_BC22', 'rs231783152_BC21', 'rs231827732_TC21', 'rs231864851_BC21', 'rs232251489_TC21', 'rs232456961_BC21', 'rs233104467_BC21', 'rs233195649_TC21', 'rs233485043_TC21', 'rs233896555_BC21', 'rs233959756_TC21', 'rs234134949_BC21', 'rs234243290_BC21', 'rs234657554_TC21', 'rs234689890_TC21', 'rs234825256_TC21', 'rs234936391_TC21', 'rs235353808_BC21', 'rs235357786_TC21', 'rs235486993_BC21', 'rs235802917_BC21', 'rs236081924_BC21', 'rs236594123_TC21', 'rs236624633_BC21', 'rs236632242_BC21', 'rs236680206_BC21', 'rs236731053_BC21', 'rs236989564_BC21', 'rs237581131_TC21', 'rs238381919_BC21', 'rs238682393_BC21', 'rs238722708_BC21', 'rs238891333_BC21', 'rs238899583_TC21', 'rs239118269_BC21', 'rs239634423_BC21', 'rs240049454_TC21', 'rs240101497_BC21', 'rs240174762_BC21', 'rs240421468_TC21', 'rs241079868_BC21', 'rs241125381_TC21', 'rs241169121_BC21', 'rs241373111_TC21', 'rs241466001_TC21', 'rs241697348_BC21', 'rs242382401_BC21', 'rs242399136_TC21', 'rs242930132_BC21', 'rs242957558_BC21', 'rs243106940_BC21', 'rs243168225_BC21', 'rs243427388_TC21', 'rs243428925_BC21', 'rs243789120_TC21', 'rs243873738_TC21', 'rs243890462_BC21', 'rs243976225_BC21', 'rs244056657_BC21', 'rs244300579_TC21', 'rs244808377_TC21', 'rs244847823_TC21', 'rs244908367_BC21', 'rs244963628_BC21', 'rs245307400_TC21', 'rs245545222_TC21', 'rs245995211_TC21', 'rs246127201_TC21', 'rs246227460_BC21', 'rs246262673_TC21', 'rs246625745_TC21', 'rs246839600_TC21', 'rs246917109_BC21', 'rs247282477_BC21', 'rs247394039_BC21', 'rs247447149_TC21', 'rs247490468_TC21', 'rs247542741_TC21', 'rs247569630_TC21', 'rs247636862_TC21', 'rs247664967_BC21', 'rs247824387_BC21', 'rs247974563_TC21', 'rs248191939_BC21', 'rs248221995_TC21', 'rs248476972_BC21', 'rs248699051_BC21', 'rs248892702_TC21', 'rs249350499_BC21', 'rs249746634_BC21', 'rs250170981_TC21', 'rs250218667_BC21', 'rs250971740_TC21', 'rs251378371_TC21', 'rs251745524_TC21', 'rs251915108_TC21', 'rs251928543_BC21', 'rs252101019_TC21', 'rs252470098_TC21', 'rs252606438_TC21', 'rs252749385_BC21', 'rs252815897_TC21', 'rs252823583_BC21', 'rs252915005_TC21', 'rs252985227_BC21', 'rs253167024_BC21', 'rs253178866_BC21', 'rs253330544_TC21', 'rs253397710_BC21', 'rs253697629_BC21', 'rs253699085_BC21', 'rs254050478_TC21', 'rs254463362_TC21', 'rs254558324_BC21', 'rs254673009_BC21', 'rs254769212_BC21', 'rs254962283_TC21', 'rs255030006_BC21', 'rs255202501_BC21', 'rs255216138_TC21', 'rs255314446_TC21', 'rs255474359_BC21', 'rs255474359_BC22', 'rs255647337_TC21', 'rs255694468_BC21', 'rs255702174_TC21', 'rs255815160_BC21', 'rs256284328_BC21', 'rs256556599_TC21', 'rs256709818_BC21', 'rs256716102_TC21', 'rs256801086_BC21', 'rs256836054_TC21', 'rs256858903_TC21', 'rs256923082_BC21', 'rs257179659_TC21', 'rs257206219_TC21', 'rs257351463_TC21', 'rs257379098_TC21', 'rs257429108_TC21', 'rs257587236_BC21', 'rs257595325_BC21', 'rs258027624_BC21', 'rs258302806_TC21', 'rs258337496_TC21', 'rs258676951_TC21', 'rs258892227_TC21', 'rs258902364_TC21', 'rs259149286_BC21', 'rs259159595_TC21', 'rs259178054_TC21', 'rs259183823_TC21', 'rs259233921_BC21', 'rs259254565_BC21', 'rs259355761_TC21', 'rs259358660_TC21', 'rs259517289_BC21', 'rs259540824_BC21', 'rs259964007_BC21', 'rs260931187_TC21', 'rs261034204_TC21', 'rs261100115_BC21', 'rs261434524_BC21', 'rs261472181_TC21', 'rs261531602_TC21', 'rs261766351_TC21', 'rs262998696_TC21', 'rs263472428_TC21', 'rs263489370_TC21', 'rs263566580_BC21', 'rs263669189_BC21', 'rs264131445_TC21', 'rs264144035_BC21', 'rs264283936_TC21', 'rs264337053_BC21', 'rs264840505_BC21', 'rs264841685_BC21', 'rs265036461_TC21', 'rs265113641_BC21', 'rs265515652_BC21', 'rs265785977_TC21', 'rs265921441_BC21', 'rs266047304_TC21', 'rs266134012_TC21', 'rs26847171_TC21', 'rs26864015_BC21', 'rs26897876_TC21', 'rs26915261_BC21', 'rs27065840_BC21', 'rs27066470_BC21', 'rs27066649_TC21', 'rs27201936_TC21', 'rs27218157_TC21', 'rs27266822_TC21', 'rs27307306_TC21', 'rs27380067_TC21', 'rs27410792_TC21', 'rs27439726_TC21', 'rs27458378_BC21', 'rs27469737_TC21', 'rs27569332_TC21', 'rs27647916_BC21', 'rs27647916_BC22', 'rs27665449_BC21', 'rs27670649_BC21', 'rs27703534_TC21', 'rs27904525_TC21', 'rs27930372_BC21', 'rs27932276_TC21', 'rs28031290_TC21', 'rs28104183_TC21', 'rs28159258_BC21', 'rs28161401_BC21', 'rs28173501_TC21', 'rs28186846_BC21', 'rs28327594_TC21', 'rs28331298_BC21', 'rs29041768_TC21', 'rs29056176_TC21', 'rs29072964_TC21', 'rs29095631_TC21', 'rs29097237_TC21', 'rs29105385_BC21', 'rs29107060_TC21', 'rs29156031_BC21', 'rs29160242_TC21', 'rs29175830_TC21', 'rs29178198_TC21', 'rs29192233_BC21', 'rs29203451_TC21', 'rs29208028_BC21', 'rs29215937_TC21', 'rs29242322_BC21', 'rs29272296_TC21', 'rs29276929_BC21', 'rs29278211_BC21', 'rs29281557_TC21', 'rs29285409_BC21', 'rs29305617_BC21', 'rs29311435_BC21', 'rs29431399_BC21', 'rs29493699_TC21', 'rs29494376_TC21', 'rs29509115_TC21', 'rs29558397_TC21', 'rs29608214_TC21', 'rs29625043_BC21', 'rs29632888_TC21', 'rs29717428_BC21', 'rs29726863_TC21', 'rs29749171_BC21', 'rs29773309_TC21', 'rs29779002_TC21', 'rs29801485_BC21', 'rs29810546_BC21', 'rs29816695_TC21', 'rs29898912_TC21', 'rs29950245_TC21', 'rs29977096_TC21', 'rs30179612_TC21', 'rs30289272_TC21', 'rs30312012_BC21', 'rs30381374_BC21', 'rs30468722_TC21', 'rs30480269_TC21', 'rs30544720_BC21', 'rs30667335_BC21', 'rs30692090_TC21', 'rs30746315_TC21', 'rs30817317_BC21', 'rs30836183_TC21', 'rs30984204_BC21', 'rs31034349_TC21', 'rs31064702_TC21', 'rs31091714_TC21', 'rs31179078_BC21', 'rs31241000_TC21', 'rs31338587_TC21', 'rs31349762_BC21', 'rs31369752_TC21', 'rs31413653_TC21', 'rs31467504_BC21', 'rs31558350_BC21', 'rs31563231_BC21', 'rs31596046_BC21', 'rs31602860_TC21', 'rs31626506_BC21', 'rs31693830_TC21', 'rs31766592_BC21', 'rs31786682_BC21', 'rs31792310_TC21', 'rs31842080_TC21', 'rs32062798_TC21', 'rs32086004_TC21', 'rs32153474_TC21', 'rs32157573_BC21', 'rs32255258_TC21', 'rs32299283_BC21', 'rs32355522_TC21', 'rs32398298_BC21', 'rs32430537_BC21', 'rs32478480_BC21', 'rs32531309_BC21', 'rs32703576_BC21', 'rs32728065_TC21', 'rs32781835_TC21', 'rs33036540_BC21', 'rs33109580_TC21', 'rs33233558_TC21', 'rs33362956_BC21', 'rs33381927_BC21', 'rs33388595_BC21', 'rs33605391_TC21', 'rs33622938_BC21', 'rs33664267_BC21', 'rs33695576_TC21', 'rs33715364_BC21', 'rs33752746_BC21', 'rs33779096_BC21', 'rs36342546_BC21', 'rs36382248_TC21', 'rs36397114_TC21', 'rs36402268_TC21', 'rs3654986_TC21', 'rs36605103_BC21', 'rs36617736_BC21', 'rs3667332_TC21', 'rs3668421_TC21', 'rs36727900_BC21', 'rs37087486_BC21', 'rs37087486_BC22', 'rs37115174_TC21', 'rs3718745_BC21', 'rs37324321_TC21', 'rs37398972_TC21', 'rs37454539_TC21', 'rs37660116_BC21', 'rs37841887_TC21', 'rs37868712_BC21', 'rs38137123_TC21', 'rs38502664_TC21', 'rs38514242_TC21', 'rs38597635_TC21', 'rs386894787_TC21', 'rs387058893_TC21', 'rs387510518_BC21', 'rs387546874_BC21', 'rs39674167_BC21', 'rs40125798_BC21', 'rs4152042_TC21', 'rs4190153_TC21', 'rs4211703_BC21', 'rs4221598_TC21', 'rs46079949_TC21', 'rs46270449_TC21', 'rs46462788_TC21', 'rs46517650_BC21', 'rs46563785_TC21', 'rs46646908_BC21', 'rs46715553_TC21', 'rs46751997_BC21', 'rs46771972_BC21', 'rs46907146_BC21', 'rs47058788_BC21', 'rs47164625_BC21', 'rs47203680_TC21', 'rs47415320_TC21', 'rs47435801_BC21', 'rs47515890_TC21', 'rs47896085_TC21', 'rs47959292_BC21', 'rs48211255_BC21', 'rs48284079_BC21', 'rs48488840_BC21', 'rs48521828_BC21', 'rs48535333_BC21', 'rs48539460_TC21', 'rs48687294_TC21', 'rs48792241_BC21', 'rs48794940_BC21', 'rs48807362_BC21', 'rs48868852_BC21', 'rs49174199_TC21', 'rs49221881_BC21', 'rs49265791_TC21', 'rs49302583_TC21', 'rs49448999_BC21', 'rs49486182_TC21', 'rs49545944_BC21', 'rs49560492_BC21', 'rs49621304_BC21', 'rs49658760_BC21', 'rs49716043_BC21', 'rs49775858_BC21', 'rs49811955_BC21', 'rs50049494_TC21', 'rs50242871_BC21', 'rs50315637_BC21', 'rs50372932_TC21', 'rs50571417_TC21', 'rs50615896_TC21', 'rs50731739_BC21', 'rs50801342_TC21', 'rs51153073_BC21', 'rs51172188_BC21', 'rs51244814_TC21', 'rs51334756_BC21', 'rs51346161_TC21', 'rs51465303_TC21', 'rs51534139_TC21', 'rs51543475_TC21', 'rs51623688_TC21', 'rs51702443_TC21', 'rs51703304_BC21', 'rs51717066_TC21', 'rs51795756_TC21', 'rs51867246_TC21', 'rs52054624_BC21', 'rs52570224_TC21', 'rs578266421_BC21', 'rs578535906_TC21', 'rs581280974_TC21', 'rs583116188_BC21', 'rs584222858_BC21', 'rs585463869_BC21', 'rs6153119_BC21', 'rs6167777_TC21', 'rs6183279_BC21']

~847 found in the strain table

"""


def test1():
    import pandas as pd
    import methylcheck
    #df = pd.read_pickle('~/methylcheck/docs/example_data/mouse/control_probes.pkl')
    #df = df['204879580038_R06C02'][['snp_beta']]
    raw = pd.read_pickle('/Volumes/LEGX/55085/55085_MURMETVEP/control_probes.pkl')
    raw = {k: v['snp_beta'] for k,v in raw.items()}
    df = pd.DataFrame(data=raw)
    print(df)
    for sample in df.columns:
        result = methylcheck.predict.infer_strain(df[[sample]])
        print(result)

def test2():
    import pandas as pd
    import methylcheck
    #df = pd.read_pickle('~/methylcheck/docs/example_data/mouse/control_probes.pkl')
    #df = df['204879580038_R06C02'][['snp_beta']]
    raw = pd.read_pickle('/Volumes/LEGX/55085/55085_MURMETVEP/control_probes.pkl')
    raw = {k: v['snp_beta'] for k,v in raw.items()}
    df = pd.DataFrame(data=raw)
    print(df)
    for sample in df.columns:
        result = methylcheck.predict.infer_strain(df[[sample]])
        print(result)

def test():
    import pandas as pd
    import methylcheck
    results = methylcheck.predict.infer_strain('/Volumes/LEGX/55085/55085_MURMETVEP/control_probes.pkl')
    print(len(results), results['204375590039_R04C02'])
