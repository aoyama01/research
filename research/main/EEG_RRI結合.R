# フォルダの指定
DIR.RRI <- "G:/Aoyama_RRI_EEG"
DIR.EEG <- "G:/Aoyama_RRI_EEG"
# 統合データの出力
DIR.OUT <- "G:/Aoyama_RRI_EEG"
#################
# ファイル名の指定
FN.RRI <- "2019A自宅.csv"
FN.EEG <- "2019A自宅睡眠段階.csv"
FN.OUT <- "EEG_RRI.csv"
###
# 脳波の計測開始日
date.eeg <- "2019-11-21"
#################
# ファイル読み込み
setwd(DIR.RRI)
TMP.RRI <- read.csv(FN.RRI,header=TRUE,skip=5)
options(digits.secs=3)
TMP.RRI$time <- as.POSIXct(TMP.RRI$time,tz="Japan")
#TMP.RRI$time.R <- TMP.RRI$time[1]+cumsum(TMP.RRI$RRI)/1000
###
setwd(DIR.EEG)
TMP.EEG <- read.csv(FN.EEG,header=TRUE,skip=0)
TMP.EEG$date.time <- as.POSIXct(paste(date.eeg,TMP.EEG$Time),tz="Japan")
N.EEG <- nrow(TMP.EEG)
i.tmp <- which(diff(TMP.EEG$date.time)<0)+1
TMP.EEG$date.time[i.tmp:N.EEG] <- TMP.EEG$date.time[i.tmp:N.EEG]+24*60*60
#######################
# 正常値の設定
RRI.max <- 1760
RRI.min <- 350
RRI.diff <- 200
#######################
RRI <- TMP.RRI$RRI
time.RRI <- TMP.RRI$time
n.RRI <- length(RRI)
D1.RRI <- c()
D2.RRI <- c()
#
tmp <- abs(diff(RRI))
D1.RRI <- c(0,tmp)
D2.RRI <- c(tmp,0)
#
time.RRI.rev <- time.RRI[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff]
RRI.rev <- RRI[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff & D2.RRI < RRI.diff]
##################
# リサンプリング
f.resamp <- 2
###
time1 <- min(time.RRI.rev)
time2 <- max(time.RRI.rev)
time.r <- seq(time1,time2,1/f.resamp) 
RRI.r <- approx(time.RRI.rev,RRI.rev,xout=time.re)$y
######################
# プロット
#plot(time.RRI,RRI,type="l",lwd=2, col=gray(0.8))
#lines(time.r,RRI.r,lwd=2, col=2)
######################
time.sub <- TMP.EEG$date.time
n.sub <- length(time.sub)-1
######################
# RRIの平均値とSD
time <- c()
meanRR <- c()
SDRR <- c()
for(i in 1:n.sub){
 time[i] <- time.sub[i+1]
 sel <- RRI.r[time.r >= time.sub[i] & time.r < time.sub[i+1]]
 meanRR[i] <- mean(sel,na.rm=TRUE)
 SDRR[i] <- sd(sel,na.rm=TRUE)
}
time <- as.POSIXct(time,origin="1970-01-01",tz="Japan")
############################
# プロット
par(mfrow=c(4,1))
plot(time,meanRR,"l",col=2)
plot(time,SDRR,"l",col=2)
plot(time,TMP.EEG$Delta_Power[-1],"l",col=4)
plot(time,TMP.EEG$Alpha_Power[-1],"l",col=4)
############################
DAT <- TMP.EEG[-1,]
DAT$meanRR <- meanRR
DAT$SDRR <- SDRR
############################
# 統合データの書き出し
setwd(DIR.OUT)
write.table(DAT,FN.OUT, sep = ",", append=FALSE, quote=FALSE, col.names=TRUE,row.names=FALSE) 