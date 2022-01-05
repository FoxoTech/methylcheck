
"""
######## ethnicity, probes and models #######
load('GR.InfiniumMethylation/20160711//EPIC/EPIC.manifest.rda')
load('GR.InfiniumMethylation/20160711//hm450//hm450.manifest.rda')
rsprobes <- intersect(names(EPIC.manifest)[grep('rs', names(EPIC.manifest))], names(hm450.manifest)[grep('rs', names(hm450.manifest))])
rsprobes <- sort(rsprobes)
ccsprobes <- colnames(SNPswitched.maf001)
ccsprobes <- sort(ccsprobes)
samples <- intersect(rownames(SNPswitched.maf001), colnames(rs.betas))
samples <- samples[!is.na(samplerace[samples])]

df.train <- cbind(t(rs.betas[rsprobes,samples]), SNPswitched.maf001[samples, ccsprobes])
df.train[is.na(df.train)] <- 0.5
library(randomForest)
fit <- randomForest(x=df.train, y=as.factor(samplerace[rownames(df.train)]), importance=TRUE, ntree=200)
load('450k/signals.dyebias/3999492009_R01C02.rda')
load('450k/signals/3999492009_R01C02.rda')
sset[rsprobes]
b <- sset[ccsprobes]$toBetaTypeIbySum(na.mask=FALSE)

a <- getBetas(sset[rsprobes], quality.mask = F, nondetection.mask=F)
b <- getBetasTypeIbySumAlleles(sset[ccsprobes], quality.mask = F, nondetection.mask = F)
ab <- c(a,b)
predict(fit, ab)

ethnicity.ccs.probes <- ccsprobes
ethnicity.rs.probes <- rsprobes
save(ethnicity.ccs.probes, file='~/tools/sesame/sesame/data/ethnicity.ccs.probes.rda')
save(ethnicity.rs.probes, file='~/tools/sesame/sesame/data/ethnicity.rs.probes.rda')
ethnicity.model <- fit
save(ethnicity.model, file='~/tools/sesame/sesame/data/ethnicity.model.rda')



                                          AMERICAN INDIAN OR ALASKA NATIVE, ASIAN,
AMERICAN INDIAN OR ALASKA NATIVE,                                         0,     3,
ASIAN,                                                                    0,   557,
BLACK OR AFRICAN AMERICAN,                                                0,     2,
NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER,                                0,     8,
WHITE,                                                                    0,    13,
                                          BLACK OR AFRICAN AMERICAN,
AMERICAN INDIAN OR ALASKA NATIVE,                                  0,
ASIAN,                                                             0,
BLACK OR AFRICAN AMERICAN,                                       833,
NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER,                         0,
WHITE,                                                            29,

                                          NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER WHITE
AMERICAN INDIAN OR ALASKA NATIVE                                                  0    19
ASIAN                                                                             0    50
BLACK OR AFRICAN AMERICAN                                                         0    39
NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER                                         0     2
WHITE                                                                             0  7097
                                          class.error
AMERICAN INDIAN OR ALASKA NATIVE          1.000000000
ASIAN                                     0.082372323
BLACK OR AFRICAN AMERICAN                 0.046910755
NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER 1.000000000
WHITE                                     0.005883177
"""
