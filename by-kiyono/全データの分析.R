######################
# f[^ÌoÍtH_
DIR.out <- "D:/Document/¤/`_°/ªÍÊ"
###########################
DIR.INFO <- "D:/Document/¤/`_°"
DIR.EEG <- "D:/Document/¤/`_°/EEG"
DIR.HR <- "D:/Document/¤/`_°/RRI"
###########################
setwd(DIR.INFO)
INFO <- na.omit(read.csv("EXP_INFO.csv",header=TRUE))
N.exp <- nrow(INFO)
###########################
setwd(DIR.EEG)
FN.EEG <- list.files(DIR.EEG,recursive=FALSE)
#data.frame(FN.EEG)
###########################
#ID.HR <- sub('\\.[^.]*', "",list.files(DIR.HR))
#ID.list <- sort(intersect(ID.EEG,ID.HR))
#n.ID <- length(ID.list)
#setdiff(ID.HR,ID.ECG)
#
iiid <- 1
for(i in 1:13){
setwd(DIR.EEG)
TMP <- read.csv(INFO[i,"SLEEP_STAGE"],header=TRUE,stringsAsFactors=FALSE)
# útðÁ¦ÄúÉÏ·
TMP$Time <- as.POSIXct(paste(INFO[i,"DATE"],TMP$Time))
n.time <- length(TMP$Time)
n.24 <- which(diff(TMP$Time)<0)+1
TMP$Time[n.24:n.time] <- TMP$Time[n.24:n.time]+24*60*60
#
T1 <- min(TMP$Time)+10*60
T2 <- max(TMP$Time)-10*60
#
DAT.STG <- TMP[TMP$Epoch != "",]
#
DAT.STG$STG <- 0
DAT.STG[DAT.STG$Score=="R","STG"] <- -1
DAT.STG[DAT.STG$Score=="N1","STG"] <- -2
DAT.STG[DAT.STG$Score=="N2","STG"] <- -3
DAT.STG[DAT.STG$Score=="N3","STG"] <- -4

DAT.STG$col <- "#fde8e8" #"#ffffd8"
DAT.STG[DAT.STG$Score=="R","col"] <- "#d8ffd8"
DAT.STG[DAT.STG$Score=="N1","col"] <- "#e6f8ff"
DAT.STG[DAT.STG$Score=="N2","col"] <- "#d4ecff"
DAT.STG[DAT.STG$Score=="N3","col"] <- "#b9e0ff"
##############################
setwd(DIR.HR)
dt <- 0.5
Q.level <- 0.5
Q.RR <- c()
t.loc <- c()
VARR <- c()
##########
logLF.m <- c()
logHF.m <- c()
### 0428ÇÁ ###
logLF.nu.m <- c()
logHF.nu.m <- c()
logLF.TP.m <- c()
logHF.TP.m <- c()
###############
LFHF.m <- c()
#####
 DAT <- read.csv(paste(INFO[i,"MyBEAT"],".csv",sep=""),header=TRUE,stringsAsFactors=FALSE,skip=5)
 DAT$time <- as.POSIXct(DAT$time)
 DAT <- DAT[DAT$time >= T1 & DAT$time <= T2,]
 time.R <- as.POSIXct(DAT$time)
 RRI <- DAT$RRI
 #######################
 time.RRI <- cumsum(RRI/1000)
 #######################
 # ³ílÌÝè
 RRI.max <- 2000
 RRI.min <- 400
 RRI.diff <- 200
 #######################
 # nñÌ·³
 n.RRI <- length(RRI)
 # ÙílÌO
 D1.RRI <- c()
 D2.RRI <- c()
 D1.RRI[1] <- 0
 D2.RRI[n.RRI] <- 0
 D1.RRI[2:n.RRI] <- abs(RRI[2:n.RRI]-RRI[1:(n.RRI-1)])
 D2.RRI[1:(n.RRI-1)] <- D1.RRI[2:n.RRI]
 time.RRI.rev <- time.RRI[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff]
# & D2.RRI < RRI.diff]
 time.R <- time.R[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff]
 RRI.rev <- RRI[RRI > RRI.min & RRI < RRI.max & D1.RRI < RRI.diff]
#           & D2.RRI < RRI.diff]

 #######################
 Q.RR[i] <- length(RRI.rev)/length(RRI)
 if(Q.RR[i] >= Q.level){
 ###
   time.sub <- seq(T1-5*60,T2+5*60,5*60)
   n.sub <- length(time.sub)-1
   LF <- c()
   HF <- c()
   ### 0428ÇÁ ###
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
   SLP.STG <- c()

   for(j in 1:n.sub){
     t.loc[j] <- time.sub[j+1]
     timeR.loc <- time.RRI.rev[time.R >= time.sub[j] & time.R < time.sub[j+1]]
     RRI.loc <- RRI.rev[time.R >= time.sub[j] & time.R < time.sub[j+1]]
     if(sum(DAT.STG[DAT.STG$Time >= time.sub[j] & DAT.STG$Time < time.sub[j+1],"Score"]=="W")>0){
       SLP.STG[j] <- "W"
     }else{
       TMP <- table(DAT.STG[DAT.STG$Time >= time.sub[j] & DAT.STG$Time < time.sub[j+1],"Score"])
       SLP.STG[j] <- names(TMP[TMP==max(TMP)])[1]
     }

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
       # p[XyNgÌSÊÏ
       psd.sum <- sum(psd$spec)
       psd_wo_VLF <- sum(psd$spec[psd$freq > 0.04])
       #####################################
       # ügÌæwW
       #@HFp[ [ms^2]
       HF[j] <- VARR[j]*sum(psd$spec[psd$freq > 0.15 & psd$freq <= 0.4])/psd.sum
       #@LFp[ [ms^2]
       LF[j] <- VARR[j]*sum(psd$spec[psd$freq > 0.04 & psd$freq <= 0.15])/psd.sum
       #@LF/HF
       LFHF[j] <- LF[j]/HF[j]
       #######################
       # 0428ÇÁ
       VLF[j] <- VARR[j]*psd_wo_VLF/psd.sum
       HF.nu[j] <- sum(psd$spec[psd$freq > 0.15 & psd$freq <= 0.4])/psd_wo_VLF*100
       LF.nu[j] <- sum(psd$spec[psd$freq > 0.04 & psd$freq <= 0.15])/psd_wo_VLF*100
       HF.TP[j] <- sum(psd$spec[psd$freq > 0.15 & psd$freq <= 0.4])/psd.sum*100
       LF.TP[j] <- sum(psd$spec[psd$freq > 0.04 & psd$freq <= 0.15])/psd.sum*100
       ##########
@@@@@}else{
       VARR[j] <- NA
       SDRR[j] <- NA
       RMSSD[j] <- NA
       pRR50[j] <- NA
       meanRR[j] <- NA 
       LF[j] <- NA
       HF[j] <- NA
       LFHF[j] <- NA
       # 0428ÇÁ
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

###################
 setwd(DIR.out)
 pdf(paste(INFO$SUBJECT[i],"_",INFO$CONDITION[i],"_HRV.pdf",sep=""),width = 12, height = 12.5)
  par(mfrow=c(4,2))
  par(mar=c(4,7,1,2))
#################
# Sleep Stage
  plot(DAT.STG$Time,DAT.STG$STG,type="l",xlim=c(T1-5*60,T2+5*60),ylim=c(-4.5,0.5),col=0,xlab="time",ylab="",las=1,cex.axis=1.4,cex.lab=1.8,yaxt="n",xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(DAT.STG$STG[k],DAT.STG$STG[k+1],-5,-5),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(DAT.STG$Time,DAT.STG$STG,type="l",xlim=c(T1-5*60,T2+5*60),ylim=c(-4.5,0.5),col=1,lwd=1,xlab="time",ylab="",las=1,cex.axis=1.4,cex.lab=1.8,yaxt="n",xaxs="i")
  abline(v=c(T1,T2),lty=2,col=4)
  mtext("Sleep Stage", side = 2, line = 4.5,cex=1.2,col=1)
  axis(side=2, at=-4:0, labels=c("SWS","N2","N1","REM","wake"),las=1,cex.axis=1.4)
#################
# meanRR
  T.HRV <- as.POSIXct(t.loc,origin="1970-1-1 00:00:00") 
  plot(T.HRV,meanRR,type="l",col=3,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(1500,1500,200,200),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
 00:00:00"),meanRR,type="l",col=1,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")

 abline(v=c(T1,T2),lty=2,col=4)
  mtext("meanRR [ms]", side = 2, line = 4.5,cex=1.2,col=1)
#################
# pRR50
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),pRR50,type="l",col=3,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
#  S.SFT <- DAT.STG[diff(DAT.STG$STG) != 0,"Time"]
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(1000,1000,0,0),col=DAT.STG$col[k+1],
border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),pRR50,type="l",col=1,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  abline(v=c(T1,T2),lty=2,col=4)
  mtext("pRR50 [%]", side = 2, line = 4.5,cex=1.2,col=1)
#################
# RMSSD
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),RMSSD,type="l",col=3,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
#  S.SFT <- DAT.STG[diff(DAT.STG$STG) != 0,"Time"]
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(1000,1000,0,0),col=DAT.STG$col[k+1],
border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),RMSSD,type="l",col=1,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  abline(v=c(T1,T2),lty=2,col=4)
  mtext("RMSSD [ms]", side = 2, line = 4.5,cex=1.2,col=1)
#################
# SDRR
  plot(T.HRV,SDRR,type="l",col=3,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(1000,1000,0,0),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1 00:00:00"),SDRR,type="l",col=1,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  abline(v=c(T1,T2),lty=2,col=4)
  mtext("SDRR [ms^2]", side = 2, line = 4.5,cex=1.2,col=1)
#################
# ln HF
  plot(T.HRV,log(HF),type="l",col=3,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(20,20,0,0),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),log(HF),type="l",col=1,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  abline(v=c(T1,T2),lty=2,col=4)
  mtext("ln HF [ln ms^2]", side = 2, line = 4.5,cex=1.2,col=1)
#################
# ln LF
  plot(T.HRV,log(LF),type="l",col=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(20,20,0,0),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),log(LF),type="l",col=1,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i")
 abline(v=c(T1,T2),lty=2,col=4)
  mtext("ln LF [ln ms^2]", side = 2, line = 4.5,cex=1.2,col=1)
#################
# LF/HF
  plot(T.HRV,LFHF,type="l",col=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,ylim=c(0,15),xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(18,18,-1,-1),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),LFHF,type="l",col=1,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,ylim=c(0,15),xaxs="i")
 abline(v=c(T1,T2),lty=2,col=4)
  mtext("LF/HF", side = 2, line = 4.5,cex=1.2,col=1)
#################
# VLF
  plot(T.HRV,log(VLF),type="l",col=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,ylim=c(0,15),xaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(18,18,-1,-1),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),log(VLF),type="l",col=1,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,ylim=c(0,15),xaxs="i")
 abline(v=c(T1,T2),lty=2,col=4)
  mtext("ln VLF [ln ms^2]", side = 2, line = 4.5,cex=1.2,col=1)
#################
# LF.nu, HF.nu
  plot(T.HRV,LF.nu,type="l",col=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylim=c(0,100),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i",yaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(101,101,-1,-1),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),LF.nu,type="l",col=2,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylim=c(0,100),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i",yaxs="i")
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),HF.nu,type="l",col=3,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylim=c(0,100),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i",yaxs="i")
 abline(v=c(T1,T2),lty=2,col=4)
  mtext("LF.nu (red), HF.nu (green) [%]", side = 2, line = 4.5,cex=1.2,col=1)
#################
# LF.TP, HF.TP
  plot(T.HRV,LF.TP,type="l",col=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylim=c(0,100),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i",yaxs="i")
  for(k in 1:length(DAT.STG$STG)){
   polygon(c(DAT.STG$Time[k],DAT.STG$Time[k+1],DAT.STG$Time[k+1],DAT.STG$Time[k]),c(100,100,0,0),col=DAT.STG$col[k+1], border=NA)
  }
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),LF.TP,type="l",col=2,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylim=c(0,100),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i",yaxs="i")
  par(new=TRUE)
  plot(as.POSIXct(t.loc,origin="1970-1-1
00:00:00"),HF.TP,type="l",col=3,lwd=2,xlab="time",xlim=c(T1-5*60,T2+5*60),ylim=c(0,100),ylab="",las=1,cex.axis=1.4,cex.lab=1.8,xaxs="i",yaxs="i")
 abline(v=c(T1,T2),lty=2,col=4)
  mtext("LF.p (red), HF.p (green) [%]", side = 2, line = 4.5,cex=1.2,col=1)
  dev.off()
 }else{
  logLF.m <- NA
  logHF.m <- NA
  LFHF.m <- NA
 }
###
DAT.OUT <- data.frame(Time=T.HRV,Stage=SLP.STG,meanRR=meanRR,RMSSD=RMSSD,pRR50=pRR50,SDRR=SDRR,lnHF=log(HF),lnLF=log(LF),LFHF,lnVLF=log(VLF),HF.nu=HF.nu,LF.nu=LF.nu,HF.p=HF.TP,LF.p=LF.TP)
write.csv(DAT.OUT,paste(INFO$SUBJECT[i],"_",INFO$CONDITION[i],"_HRV.csv",sep=""),row.names = FALSE)
#data.frame(subject=SBJ[i],qualRRI=Q.RR[i],logHF=logHF.m,logLF=logLF.m,LFHF=LFHF.m)
}



