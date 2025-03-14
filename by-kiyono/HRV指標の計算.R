DIR.RRI <- "D:/Documents/研究/緒形_睡眠/RRI"
###
# ファイル名
FN <- "A-homelab(20201203-08).csv"
# 日付
DATE <- "2020-12-03"
DATE <- as.Date(DATE,tz="Japan")
###
setwd(DIR.RRI)
###
dt <- 0.5
Q.level <- 0.5
Q.RR <- c()
t.loc <- c()
VARR <- c()
##########
logLF.m <- c()
logHF.m <- c()
### 0428追加 ###
logLF.nu.m <- c()
logHF.nu.m <- c()
logLF.TP.m <- c()
logHF.TP.m <- c()
###############
LFHF.m <- c()
Q.level <- 0.8

 ###
 TMP <- read.csv(FN,header=TRUE,stringsAsFactors=FALSE,skip=5)
 TMP$time <- as.POSIXct(TMP$time,tz="Japan")
 TMP$date <- as.Date(TMP$time,tz="Japan")
 # 指定した日にちを選択
 DAT <- TMP[TMP$date==DATE,]
 time.R <- as.POSIXct(DAT$time,tz="Japan")

 T1 <- as.POSIXct(format(time.R[1],"%Y-%m-%d %H:%M:00"),tz="Japan")
 T2 <- as.POSIXct(format(tail(time.R,1),"%Y-%m-%d %H:%M:00"),tz="Japan")
 RRI <- DAT$RRI
 #######################
 time.RRI <- cumsum(RRI/1000)
 #######################
 # 正常値の設定
 RRI.max <- 2000
 RRI.min <- 320
 RRI.diff <- 200
 #######################
 # 時系列の長さ
 n.RRI <- length(RRI)
 # 異常値の除外
 D1.RRI <- c()
 D2.RRI <- c()
 D1.RRI[1] <- 0
 D1.RRI[2:n.RRI] <- abs(RRI[2:n.RRI]-RRI[1:(n.RRI-1)])
 time.RRI.rev <- time.RRI[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff]
 time.R <- time.R[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff]
 RRI.rev <- RRI[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff]
 #######################
 Q.RR[i] <- length(RRI.rev)/length(RRI)
 if(Q.RR[i] >= Q.level){
 ###
   time.sub <- seq(T1-5*60,T2+5*60,5*60)
   n.sub <- length(time.sub)-1
   LF <- c()
   HF <- c()
   ### 0428追加 ###
   LF.nu <- c()
   HF.nu <- c()
   LF.TP <- c()
   HF.TP <- c()
   VLF <- c()
   ###############
   LFHF <- c()
   SDRR <- c()
   meanRR <- c()
   RMSSD <- c()
   pRR50 <- c()

   DATE <- as.Date(as.POSIXct(T1,origin="1970-1-1 00:00:00"),tz="Japan") #t.loc[1]

   for(j in 1:n.sub){
     t.loc[j] <- time.sub[j+1]
     timeR.loc <- time.RRI.rev[time.R >= time.sub[j] & time.R < time.sub[j+1]]
     RRI.loc <- RRI.rev[time.R >= time.sub[j] & time.R < time.sub[j+1]]

     if(sum(RRI.loc) >= 10000*Q.level){
       VARR[j] <- var(RRI.loc,na.rm=TRUE)
       DRR <- diff(RRI.loc,na.rm=TRUE)
       RMSSD[j] <-  sqrt(mean(DRR^2))
       pRR50[j] <- sum(abs(DRR)>50)/length(DRR)*100
       SDRR[j] <- sqrt(VARR[j])
       meanRR[j] <- mean(RRI.loc,na.rm=TRUE)
       t.resamp <- seq(timeR.loc[1],max(timeR.loc),dt)
       RRI.r <- approx(timeR.loc,RRI.loc,xout=t.resamp,method = "linear")$y
       psd <- spectrum(RRI.r,plot=FALSE)
       psd$freq <- psd$freq/dt
       #####################################
       # パワースペクトルの全面積
       psd.sum <- sum(psd$spec)
       psd_wo_VLF <- sum(psd$spec[psd$freq > 0.04])
       #####################################
       # 周波数領域指標
       #　HFパワー [ms^2]
       HF[j] <- VARR[j]*sum(psd$spec[psd$freq > 0.15 & psd$freq <= 0.4])/psd.sum
       #　LFパワー [ms^2]
       LF[j] <- VARR[j]*sum(psd$spec[psd$freq > 0.04 & psd$freq <= 0.15])/psd.sum
       #　LF/HF
       LFHF[j] <- LF[j]/HF[j]
       #######################
       # 0428追加
       VLF[j] <- VARR[j]*psd_wo_VLF/psd.sum
       HF.nu[j] <- sum(psd$spec[psd$freq > 0.15 & psd$freq <= 0.4])/psd_wo_VLF*100
       LF.nu[j] <- sum(psd$spec[psd$freq > 0.04 & psd$freq <= 0.15])/psd_wo_VLF*100
       HF.TP[j] <- sum(psd$spec[psd$freq > 0.15 & psd$freq <= 0.4])/psd.sum*100
       LF.TP[j] <- sum(psd$spec[psd$freq > 0.04 & psd$freq <= 0.15])/psd.sum*100
       ##########
　　　　　}else{
       VARR[j] <- NA
       SDRR[j] <- NA
       RMSSD[j] <- NA
       pRR50[j] <- NA
       meanRR[j] <- NA 
       LF[j] <- NA
       HF[j] <- NA
       LFHF[j] <- NA
       # 0428追加
       VLF[j] <- NA
       HF.nu[j] <- NA
       LF.nu[j] <- NA
       HF.TP[j] <- NA
       LF.TP[j] <- NA
       ##########

     }
   }
 ###
  logLF.m <- mean(log(LF[LF>0 & HF>0]),na.rm=TRUE)
  logHF.m <- mean(log(HF[LF>0 & HF>0]),na.rm=TRUE)
  LFHF.m <- mean(LFHF[LF>0 & HF>0],na.rm=TRUE)
  }

